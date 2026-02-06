"""
Docstring for src.sanction_parser.pipelines.resolver
Entity Resolution engine for processed JSONL records.
- Loads all *.jsonl / *.jsonl.gz from settings.PROCESSED_DIR (recursively)
- Builds similarity graph using blocking + fuzzy rules
- Splits clusters by DOB-years

"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from metaphone import doublemetaphone
from rapidfuzz import fuzz

from amani_aml.core.config import settings
from amani_aml.utils.nat_to_iso import (
    normalize_country_fields_in_profile,
    normalize_nationalities,
)

logger = logging.getLogger("sanction_parser.entity_resolver")


class EntityResolver:
    """
    Entity Resolution engine for processed JSONL records.
    - Loads all *.jsonl / *.jsonl.gz from settings.PROCESSED_DIR (recursively)
    - Builds similarity graph using blocking + fuzzy rules
    - Splits clusters by DOB-years
    - Merges to golden profiles and saves JSONL.GZ + metadata
    """

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        output_file: str = "Golden_Export.jsonl",
        max_block_size: int = 500,
        debug_top_n_clusters: int = 20,
        provider: str = "AmaniAI",
    ):
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROCESSED_DIR
        self.output_dir = Path(output_dir) if output_dir else (settings.DATA_LAKE_DIR / "new_data")
        self.output_file = output_file
        self.max_block_size = max_block_size
        self.debug_top_n_clusters = debug_top_n_clusters
        self.provider = provider
        self.state_file = self.output_dir / "resolution_state_hashes.json"

    # ---------------------------------------------------------------------
    # Public entrypoint
    # ---------------------------------------------------------------------
    def run(self) -> None:
        logger.info("=== ROBUST ENTITY RESOLUTION ===")
        logger.info("Processed dir: %s", self.processed_dir)
        logger.info("Output dir: %s", self.output_dir)

        all_profiles = self._load_all_profiles()
        logger.info("Loaded %d profiles.", len(all_profiles))

        if not all_profiles:
            logger.warning("No profiles loaded. Exiting.")
            return

        graph = self._build_similarity_graph(all_profiles)
        clusters = list(nx.connected_components(graph))
        logger.info("Resolved into %d connected components (candidate entities).", len(clusters))

        merged_results: List[Dict[str, Any]] = []
        split_total = 0

        for component in clusters:
            cluster_profiles = [all_profiles[i] for i in component]
            subclusters = self._split_cluster_by_dob(cluster_profiles)
            split_total += len(subclusters)

            for sub in subclusters:
                merged_results.append(self._merge_cluster(sub))

        logger.info("After DOB-aware split: %d final entity clusters.", split_total)

        self._save_final_output(merged_results)
        #self._debug_largest_clusters_with_split(all_profiles, clusters)
        

    def _debug_largest_clusters_with_split(
        self, all_profiles: List[Dict[str, Any]], clusters: List[Set[int]]
    ) -> None:
        clusters_sorted = sorted(clusters, key=len, reverse=True)
        top_n = min(self.debug_top_n_clusters, len(clusters_sorted))

        logger.info(
            "--- TOP %d LARGEST CONNECTED COMPONENTS (Before DOB split) ---",
            top_n,
        )

        for i in range(top_n):
            c_indices = list(clusters_sorted[i])

            names: List[str] = []
            sources: Set[str] = set()

            for idx in c_indices:
                if not isinstance(all_profiles[idx], dict):
                    continue

                d = all_profiles[idx].get("data") or {}
                data_sources = all_profiles[idx].get("datasets") or []

                names.append(d.get("fullName") or "")
                for s in data_sources:
                    if isinstance(s, str) and s.strip():
                        sources.add(s)

            logger.info(
                "Component %d (%d profiles): names=%s | sources=%s",
                i + 1,
                len(c_indices),
                names,
                sorted(sources),
            )

            cluster_profiles = [all_profiles[idx] for idx in c_indices]
            subclusters = self._split_cluster_by_dob(cluster_profiles)

            logger.info("  -> DOB-aware split into %d subclusters:", len(subclusters))

            for j, sub in enumerate(subclusters, 1):
                sub_names: List[str] = []
                sub_years: Set[str] = set()
                sub_sources: Set[str] = set()

                for p in sub:
                    if not isinstance(p, dict):
                        continue

                    d = p.get("data", {}) or {}
                    sub_names.append(d.get("fullName") or "")
                    sub_years.update(self._extract_years(d.get("datesOfBirthIso") or []))

                    for s in (p.get("datasets") or []):
                        if isinstance(s, str) and s.strip():
                            sub_sources.add(s)

                logger.info(
                    "     - Subcluster %d: size=%d years=%s names=%s sources=%s",
                    j,
                    len(sub),
                    sorted(sub_years),
                    sub_names,
                    sorted(sub_sources),
                )

    # ---------------------------------------------------------------------
    # Load JSONL / JSONL.GZ
    # ---------------------------------------------------------------------
    def _iter_input_files(self) -> List[Path]:
        if not self.processed_dir.exists():
            return []
        files = list(self.processed_dir.rglob("*.jsonl")) + list(self.processed_dir.rglob("*.jsonl.gz"))
        return sorted(files)

    def _load_all_profiles(self) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []

        files = self._iter_input_files()
        logger.info("Found %d input files under '%s'.", len(files), self.processed_dir)

        for fpath in files:
            try:
                if fpath.name.endswith(".gz"):
                    opener = lambda: gzip.open(fpath, "rt", encoding="utf-8")  # noqa: E731
                else:
                    opener = lambda: fpath.open("r", encoding="utf-8")  # noqa: E731

                with opener() as f:
                    for line_number, line in enumerate(f, 1):
                        line = (line or "").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if not isinstance(obj, dict):
                                logger.warning("Skipping non-object JSON at %s:%d", fpath, line_number)
                                continue

                            d = obj.get("data")
                            if isinstance(d, dict):
                                # Ensure nationality fields are normalized as ISO2
                                normalize_country_fields_in_profile(d)

                            profiles.append(obj)

                        except json.JSONDecodeError as e:
                            logger.warning("Skipping invalid JSON at %s:%d -> %s", fpath, line_number, e)

            except Exception as e:
                logger.exception("Error reading file %s: %s", fpath, e)

        return profiles

    # ---------------------------------------------------------------------
    # Text normalization helpers
    # ---------------------------------------------------------------------
    _UNICODE_APOS_RE = re.compile(r"[’‘`´]")
    _DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+")
    _NONALNUM_RE = re.compile(r"[^\w\s]", re.UNICODE)
    _WS_RE = re.compile(r"\s+", re.UNICODE)

    _NAME_STOPWORDS = {
        "al",
        "el",
        "as",
        "ash",
        "ad",
        "at",
        "ar",
        "az",
        "abu",
        "umm",
        "um",
        "bin",
        "ibn",
        "bint",
        "ben",
        "b",
        "st",
        "saint",
        "de",
        "del",
        "da",
        "di",
        "la",
        "le",
        "the",
        "of",
        "abd",
        "abdel",
        "abdul",
    }

    def _norm_text(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        if not s:
            return ""
        s = self._UNICODE_APOS_RE.sub("'", s)
        s = self._DASH_RE.sub(" ", s)
        s = s.lower()
        s = self._NONALNUM_RE.sub(" ", s)
        s = self._WS_RE.sub(" ", s).strip()
        return s

    def _name_tokens(self, name: str) -> List[str]:
        tokens = self._norm_text(name).split()
        return [t for t in tokens if t]

    def _pick_first_meaningful_token(self, tokens: List[str]) -> str:
        for t in tokens:
            if t not in self._NAME_STOPWORDS and len(t) >= 2:
                return t
        return tokens[0] if tokens else ""

    def _pick_last_meaningful_token(self, tokens: List[str]) -> str:
        for t in reversed(tokens):
            if t not in self._NAME_STOPWORDS and len(t) >= 2:
                return t
        return tokens[-1] if tokens else ""

    def _metaphone(self, token: str) -> str:
        if not token:
            return ""
        try:
            return doublemetaphone(token)[0] or ""
        except Exception:
            return ""

    # ---------------------------------------------------------------------
    # Blocking utilities
    # ---------------------------------------------------------------------
    def _fingerprint_name(self, name: str) -> str:
        if not name:
            return ""
        tokens = sorted(set(self._name_tokens(name)))
        return " ".join(tokens)

    def _extract_years(self, dobs: Any) -> Set[str]:
        years: Set[str] = set()
        if not dobs or not isinstance(dobs, list):
            return years

        for dob in dobs:
            if not isinstance(dob, str):
                continue
            s = dob.strip()
            if not s:
                continue
            for y in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", s):
                years.add(y)

        return years

    def _get_blocking_keys(self, profile: Dict[str, Any]) -> Set[str]:
        keys: Set[str] = set()
        data = profile.get("data", {}) if isinstance(profile, dict) else {}

        full_name = data.get("fullName") or ""
        if not isinstance(full_name, str) or not full_name.strip():
            return keys

        fp = self._fingerprint_name(full_name)
        if fp:
            keys.add(f"fp:{fp}")

        years = self._extract_years(data.get("datesOfBirthIso") or [])
        tokens = self._name_tokens(full_name)

        first_tok = self._pick_first_meaningful_token(tokens)
        last_tok = self._pick_last_meaningful_token(tokens)

        if years and first_tok:
            ph = self._metaphone(first_tok)
            if ph:
                for y in years:
                    keys.add(f"ph:{ph}:{y}")

        if years and last_tok:
            phl = self._metaphone(last_tok)
            if phl:
                for y in years:
                    keys.add(f"phl:{phl}:{y}")

        if years and len(tokens) >= 2:
            t2 = tokens[-2:]
            t2 = [t for t in t2 if t not in self._NAME_STOPWORDS]
            if len(t2) >= 2:
                a, b = sorted(t2[-2:])
                if len(a) >= 2 and len(b) >= 2:
                    for y in years:
                        keys.add(f"tail2:{a}_{b}:{y}")

        fp_tokens = fp.split()
        long_tokens = [t for t in fp_tokens if len(t) > 5]

        if years:
            for token in long_tokens:
                keys.add(f"tok:{token}")
        else:
            if len(long_tokens) >= 2:
                for token in long_tokens:
                    keys.add(f"tok:{token}")

        alias_candidates: List[str] = []
        for obj in (data.get("nameAliases") or []):
            if isinstance(obj, dict):
                s = obj.get("fullName") or ""
                if isinstance(s, str) and s.strip():
                    alias_candidates.append(s)
            elif isinstance(obj, str) and obj.strip():
                alias_candidates.append(obj)

        norm_aliases = []
        seen = set()
        for s in alias_candidates:
            ns = self._norm_text(s)
            if not ns:
                continue
            if len(ns.split()) < 2 and len(ns) < 10:
                continue
            if ns in seen:
                continue
            seen.add(ns)
            norm_aliases.append(ns)

        norm_aliases = sorted(norm_aliases, key=len, reverse=True)[:6]

        for ns in norm_aliases:
            afp = " ".join(sorted(set(ns.split())))
            if afp:
                keys.add(f"afp:{afp}")

            a_tokens = ns.split()
            a_first = self._pick_first_meaningful_token(a_tokens)
            a_last = self._pick_last_meaningful_token(a_tokens)

            if years:
                if a_first:
                    aph = self._metaphone(a_first)
                    if aph:
                        for y in years:
                            keys.add(f"aph:{aph}:{y}")

                if a_last:
                    aphl = self._metaphone(a_last)
                    if aphl:
                        for y in years:
                            keys.add(f"aphl:{aphl}:{y}")

                if len(a_tokens) >= 2:
                    at2 = a_tokens[-2:]
                    at2 = [t for t in at2 if t not in self._NAME_STOPWORDS]
                    if len(at2) >= 2:
                        aa, bb = sorted(at2[-2:])
                        if len(aa) >= 2 and len(bb) >= 2:
                            for y in years:
                                keys.add(f"atail2:{aa}_{bb}:{y}")

        return keys

    # ---------------------------------------------------------------------
    # Graph build (multi-year DOB down-blocking supported)
    # ---------------------------------------------------------------------
    def _build_similarity_graph(self, profiles: List[Dict[str, Any]]) -> nx.Graph:
        G = nx.Graph()
        for i in range(len(profiles)):
            G.add_node(i)

        start_time = time.time()

        blocks: Dict[str, List[int]] = {}
        logger.info("Generating blocking keys...")
        for i, p in enumerate(profiles):
            for k in self._get_blocking_keys(p):
                blocks.setdefault(k, []).append(i)

        logger.info("Generated %d blocks. Starting comparisons...", len(blocks))

        checked_pairs: Set[Tuple[int, int]] = set()
        actual_comparisons = 0
        downblocked_blocks = 0

        def _years_or_nodob(idx: int) -> List[str]:
            d = profiles[idx].get("data", {}) if isinstance(profiles[idx], dict) else {}
            years = sorted(self._extract_years(d.get("datesOfBirthIso") or []))
            return years if years else ["NODOB"]

        def _last_token_metaphone(idx: int) -> str:
            d = profiles[idx].get("data", {}) if isinstance(profiles[idx], dict) else {}
            name = d.get("fullName") or ""
            tokens = self._name_tokens(name)
            last_tok = self._pick_last_meaningful_token(tokens)
            return self._metaphone(last_tok) or ""

        for key, indices in blocks.items():
            if len(indices) < 2:
                continue

            # Down-block large buckets for certain key types
            if len(indices) > self.max_block_size:
                if key.startswith(("fp:", "afp:", "tok:")):
                    downblocked_blocks += 1
                    subblocks: Dict[str, List[int]] = {}

                    for idx in indices:
                        years_list = _years_or_nodob(idx)
                        ph_last = _last_token_metaphone(idx) or "NOPH"

                        for y in years_list:
                            subk = f"{key}|y:{y}|phl:{ph_last}"
                            subblocks.setdefault(subk, []).append(idx)

                    for _, subidxs in subblocks.items():
                        if len(subidxs) < 2:
                            continue
                        if len(subidxs) > self.max_block_size:
                            continue

                        for a in range(len(subidxs)):
                            for b in range(a + 1, len(subidxs)):
                                idx_a, idx_b = subidxs[a], subidxs[b]
                                if idx_a > idx_b:
                                    idx_a, idx_b = idx_b, idx_a

                                pair = (idx_a, idx_b)
                                if pair in checked_pairs:
                                    continue
                                checked_pairs.add(pair)

                                actual_comparisons += 1
                                if self._is_match(profiles[idx_a], profiles[idx_b]):
                                    G.add_edge(idx_a, idx_b)
                continue

            # Regular within-block comparisons
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    idx_a, idx_b = indices[a], indices[b]
                    if idx_a > idx_b:
                        idx_a, idx_b = idx_b, idx_a

                    pair = (idx_a, idx_b)
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    actual_comparisons += 1
                    if self._is_match(profiles[idx_a], profiles[idx_b]):
                        G.add_edge(idx_a, idx_b)

        duration = time.time() - start_time

        n = len(profiles)
        naive_comparisons = (n * (n - 1)) // 2
        reduction = (1 - (actual_comparisons / naive_comparisons)) * 100 if naive_comparisons else 0.0

        logger.info("========================================")
        logger.info("BENCHMARK REPORT (%d Profiles)", n)
        logger.info("Time Taken:           %.4f seconds", duration)
        logger.info("Naive Comparisons:    %s", f"{naive_comparisons:,}")
        logger.info("Blocking Comparisons: %s", f"{actual_comparisons:,}")
        logger.info("Work Saved:           %.2f%%", reduction)
        logger.info("Down-blocked blocks:  %s", f"{downblocked_blocks:,}")
        logger.info("========================================")

        return G

    # ---------------------------------------------------------------------
    # Evidence helpers
    # ---------------------------------------------------------------------
    def _collect_nationalities(self, d: Dict[str, Any]) -> Set[str]:
        raw: List[Any] = []
        raw.extend(d.get("nationalitiesIsoCodes") or [])
        raw.extend(d.get("nationality") or [])
        iso = normalize_nationalities(raw)
        return {x.upper() for x in iso if isinstance(x, str) and x.strip()}

    def _collect_aliases(self, d: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        fn = d.get("fullName")
        nfn = self._norm_text(fn) if fn else ""
        if nfn:
            out.add(nfn)

        for obj in (d.get("nameAliases") or []):
            if isinstance(obj, dict):
                n = self._norm_text(obj.get("fullName") or "")
                if n:
                    out.add(n)
            elif isinstance(obj, str):
                n = self._norm_text(obj)
                if n:
                    out.add(n)

        return out

    # ---------------------------------------------------------------------
    # Match logic
    # ---------------------------------------------------------------------
    def _is_match(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        d1 = p1.get("data", {}) if isinstance(p1, dict) else {}
        d2 = p2.get("data", {}) if isinstance(p2, dict) else {}

        name1 = self._norm_text(d1.get("fullName") or "")
        name2 = self._norm_text(d2.get("fullName") or "")
        name_score = fuzz.token_sort_ratio(name1, name2)

        dob1 = self._extract_years(d1.get("datesOfBirthIso") or [])
        dob2 = self._extract_years(d2.get("datesOfBirthIso") or [])
        has_dob1 = bool(dob1)
        has_dob2 = bool(dob2)

        nats1 = self._collect_nationalities(d1)
        nats2 = self._collect_nationalities(d2)
        nat_overlap = bool(nats1 and nats2 and (not nats1.isdisjoint(nats2)))

        dob_name_threshold = 92 if not nat_overlap else 90
        strong_name_threshold = 97 if not nat_overlap else 96
        alias_name_threshold = 93 if not nat_overlap else 92

        if has_dob1 and has_dob2:
            if dob1.isdisjoint(dob2):
                return False
            return name_score > dob_name_threshold

        a1 = self._collect_aliases(d1)
        a2 = self._collect_aliases(d2)
        alias_overlap = bool(a1 and a2 and (not a1.isdisjoint(a2)))

        if name_score >= strong_name_threshold:
            return True
        if alias_overlap and name_score >= alias_name_threshold:
            return True

        return False

    # ---------------------------------------------------------------------
    # DOB-aware splitting
    # ---------------------------------------------------------------------
    def _split_cluster_by_dob(self, profiles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        with_dob: List[Tuple[Dict[str, Any], Set[str]]] = []
        no_dob: List[Dict[str, Any]] = []

        for p in profiles:
            d = p.get("data", {}) if isinstance(p, dict) else {}
            years = self._extract_years(d.get("datesOfBirthIso") or [])
            if years:
                with_dob.append((p, years))
            else:
                no_dob.append(p)

        if not with_dob:
            return [profiles]

        buckets: List[Tuple[Set[str], List[Dict[str, Any]]]] = []
        for p, years in with_dob:
            placed = False
            for b_years, b_items in buckets:
                if not b_years.isdisjoint(years):
                    b_years.update(years)
                    b_items.append(p)
                    placed = True
                    break
            if not placed:
                buckets.append((set(years), [p]))

        def bucket_rep_name(items: List[Dict[str, Any]]) -> str:
            votes: Dict[str, int] = {}
            for it in items:
                d = it.get("data", {}) if isinstance(it, dict) else {}
                fn = d.get("fullName")
                if isinstance(fn, str) and fn.strip():
                    clean = " ".join(fn.split()).strip()
                    votes[clean] = votes.get(clean, 0) + 1
            if not votes:
                return ""
            return sorted(votes.keys(), key=lambda x: (votes[x], len(x)), reverse=True)[0]

        def strong_attach(no_dob_profile: Dict[str, Any], bucket_items: List[Dict[str, Any]]) -> Tuple[bool, int]:
            d = no_dob_profile.get("data", {}) if isinstance(no_dob_profile, dict) else {}
            p_name = (d.get("fullName") or "")
            if not isinstance(p_name, str) or not p_name.strip():
                return (False, 0)

            rep = bucket_rep_name(bucket_items)
            if not rep:
                return (False, 0)

            score = fuzz.token_sort_ratio(p_name.lower(), rep.lower())

            a_p = self._collect_aliases(d)
            a_b: Set[str] = set()
            for it in bucket_items:
                it_d = it.get("data", {}) if isinstance(it, dict) else {}
                a_b |= self._collect_aliases(it_d)
            alias_overlap = bool(a_p and a_b and (not a_p.isdisjoint(a_b)))

            if score >= 98:
                return (True, score)
            if alias_overlap and score >= 95:
                return (True, score)

            return (False, score)

        groups: List[List[Dict[str, Any]]] = [items for _, items in buckets]

        remaining_no_dob: List[Dict[str, Any]] = []
        for p in no_dob:
            best_idx = -1
            best_score = -1

            for i, bucket_items in enumerate(groups):
                ok, score = strong_attach(p, bucket_items)
                if ok and score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                groups[best_idx].append(p)
            else:
                remaining_no_dob.append(p)

        if remaining_no_dob:
            groups.append(remaining_no_dob)

        return groups

    def _compute_profile_hash(self, out_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates a deterministic MD5 hash based strictly on BUSINESS DATA.
        Uses an ALLOWLIST to ignore all metadata, timestamps, or system IDs.
        
        This hash is used for Delta detection (Change Data Capture).
        """
        data = out_obj.get("data", {})
        datasets = out_obj.get("datasets", [])

        # 1. DEFINE THE ALLOWLIST
        # Only fields in this set contribute to the hash.
        BUSINESS_KEYS = {
            # --- Identity ---
            "firstName", "middleName", "lastName", "fullName",
            "gender", "isDeceased", "isDeleted",
            "nameAliases", "aliases",
            "datesOfBirthIso", "datesOfBirthParsed",
            "datesOfDeathIso",
            "nationalitiesIsoCodes", "nationality",
            "profileImages",
            
            # --- Contact / Locality ---
            "addresses", "contactEntries",
            
            # --- Risk & Crime Data ---
            "sanEntries",
            "pepEntries",
            "relEntries", "rreEntries", "poiEntries",
            "insEntries", "ddEntries", "griEntries",
            "pepByAssociationEntries",
            
            # --- Evidence & Context ---
            "notes",
            "evidences",
            "identifiers",
            
            # --- Links ---
            "individualLinks", "businessLinks"
        }

        # Keys to strip from nested dictionaries (Metadata inside business objects)
        VOLATILE_INNER_KEYS = {"captureDateIso", "scraped_at", "originalUrl", "assetUrl"}

        # 2. Recursive Cleaner
        # - Keeps only non-volatile keys
        # - Converts SETS to sorted LISTS (Crucial: sets are not JSON serializable)
        def recursive_clean(obj):
            if isinstance(obj, dict):
                return {
                    k: recursive_clean(v) 
                    for k, v in obj.items() 
                    if k not in VOLATILE_INNER_KEYS
                }
            elif isinstance(obj, list):
                return [recursive_clean(x) for x in obj]
            elif isinstance(obj, set):
                # SAFETY FIX: Sets are unordered and not valid JSON. 
                # Convert to sorted list immediately.
                return [recursive_clean(x) for x in sorted(list(obj), key=str)]
            return obj

        # 3. Build the Hashable Structure
        hashable_root = {}
        
        # A. Add Business Data (Filtered by Allowlist)
        for k, v in data.items():
            if k in BUSINESS_KEYS:
                hashable_root[k] = recursive_clean(v)
        
        # B. Add Datasets (Crucial: If a person moves from "Warning" to "Sanction" list, hash must change)
        # We prefix with underscore to avoid collision with data keys
        hashable_root["_datasets"] = sorted(list(datasets))

        # 4. Recursive Sorter (Determinism)
        # JSON is unordered. We must enforce strict ordering of keys and lists
        # to ensure the same data always results in the exact same hash.
        def recursive_sort(obj):
            if isinstance(obj, dict):
                # Sort dictionaries by key
                return {k: recursive_sort(v) for k, v in sorted(obj.items())}
            if isinstance(obj, list):
                # Sort lists by their JSON string representation
                return sorted(
                    [recursive_sort(x) for x in obj], 
                    key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)
                )
            return obj

        hashable_sorted = recursive_sort(hashable_root)

        # 5. Hash Generation
        # dump to string -> encode to bytes -> md5 -> hex string
        json_str = json.dumps(hashable_sorted, sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.md5(json_str.encode("utf-8")).hexdigest()
        
        return content_hash
    
    
    # ---------------------------------------------------------------------
    # Merge cluster into Golden Record (keeps your schema shape)
    # ---------------------------------------------------------------------
    def _merge_cluster(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)

        merged: Dict[str, Any] = {
            "qrCode": "",
            "resourceUri": "",
            "resourceId": "",
            "firstName": "",
            "fullName": "",
            "middleName": "",
            "lastName": "",
            "gender": None,
            "provider": self.provider,
            "provider_version": now_ms,
            "isDeleted": False,
            "isDeceased": False,
            "nationality": [],
            "datesOfBirthParsed": [],
            "nameAliases": set(),
            "aliases": [],
            "datesOfBirthIso": set(),
            "datesOfDeathIso": set(),
            "nationalitiesIsoCodes": set(),
            "addresses": [],
            "profileImages": set(),
            "notes": [],
            "contactEntries": [],
            "identifiers": [],
            "evidences": [],
            "sanEntries": {"current": [], "former": []},
            "pepEntries": {"current": [], "former": []},
            "pepByAssociationEntries": [],
            "relEntries": [],
            "rreEntries": [],
            "poiEntries": [],
            "insEntries": [],
            "ddEntries": [],
            "individualLinks": [],
            "businessLinks": [],
            "griEntries": [],
        }

        final_datasets_list: Set[str] = set()

        full_name_votes: Dict[str, int] = {}
        first_votes: Dict[str, int] = {}
        middle_votes: Dict[str, int] = {}
        last_votes: Dict[str, int] = {}
        gender_votes: Dict[Any, int] = {}

        all_struct_aliases: List[Any] = []
        all_addresses: List[Any] = []
        all_dob_parsed: List[Any] = []

        gender_values: Set[Any] = set()
        dob_year_values: Set[str] = set()

        def _vote(votes: Dict[str, int], value: Optional[str]) -> None:
            if not value:
                return
            clean = " ".join(str(value).split()).strip()
            if clean:
                votes[clean] = votes.get(clean, 0) + 1

        def _extend_list(dst: List[Any], src: Any) -> None:
            if not src:
                return
            if isinstance(src, list):
                dst.extend(src)
            else:
                dst.append(src)

        for p in profiles:
            d = p.get("data", {}) if isinstance(p, dict) else {}
            final_datasets_list.update(p.get("datasets") or [])

            _vote(full_name_votes, d.get("fullName"))
            _vote(first_votes, d.get("firstName"))
            _vote(middle_votes, d.get("middleName"))
            _vote(last_votes, d.get("lastName"))

            if d.get("fullName"):
                clean_full = " ".join(str(d.get("fullName")).split()).strip()
                if clean_full:
                    merged["nameAliases"].add(clean_full)

            for alias_obj in (d.get("nameAliases") or []):
                if isinstance(alias_obj, dict):
                    a_name = alias_obj.get("fullName")
                    if a_name:
                        clean_a = " ".join(str(a_name).split()).strip()
                        if clean_a:
                            merged["nameAliases"].add(clean_a)
                elif isinstance(alias_obj, str):
                    clean_a = " ".join(alias_obj.split()).strip()
                    if clean_a:
                        merged["nameAliases"].add(clean_a)

            if d.get("gender") is not None:
                g = d.get("gender")
                gender_votes[g] = gender_votes.get(g, 0) + 1
                gender_values.add(g)

            dob_year_values.update(self._extract_years(d.get("datesOfBirthIso") or []))

            _extend_list(all_struct_aliases, d.get("aliases"))

            if d.get("isDeceased"):
                merged["isDeceased"] = True
            if d.get("isDeleted"):
                merged["isDeleted"] = True

            merged["datesOfBirthIso"].update(d.get("datesOfBirthIso") or [])
            merged["datesOfDeathIso"].update(d.get("datesOfDeathIso") or [])

            merged["nationalitiesIsoCodes"].update(normalize_nationalities(d.get("nationalitiesIsoCodes")))
            merged["nationalitiesIsoCodes"].update(normalize_nationalities(d.get("nationality")))

            merged["profileImages"].update(d.get("profileImages") or [])

            _extend_list(all_addresses, d.get("addresses"))
            _extend_list(all_dob_parsed, d.get("datesOfBirthParsed"))

            _extend_list(merged["notes"], d.get("notes"))
            _extend_list(merged["contactEntries"], d.get("contactEntries"))
            _extend_list(merged["identifiers"], d.get("identifiers"))
            _extend_list(merged["evidences"], d.get("evidences"))

            if d.get("sanEntries"):
                se = d.get("sanEntries") or {}
                merged["sanEntries"]["current"].extend(se.get("current") or [])
                merged["sanEntries"]["former"].extend(se.get("former") or [])

            if d.get("pepEntries"):
                pe = d.get("pepEntries") or {}
                merged["pepEntries"]["current"].extend(pe.get("current") or [])
                merged["pepEntries"]["former"].extend(pe.get("former") or [])

            for key in [
                "relEntries",
                "rreEntries",
                "poiEntries",
                "insEntries",
                "ddEntries",
                "pepByAssociationEntries",
                "individualLinks",
                "businessLinks",
                "griEntries",
            ]:
                _extend_list(merged[key], d.get(key))

        def _pick_best(votes: Dict[str, int]) -> str:
            if not votes:
                return ""
            sorted_vals = sorted(votes.keys(), key=lambda x: (votes[x], len(x)), reverse=True)
            return sorted_vals[0]

        merged["fullName"] = _pick_best(full_name_votes)
        merged["firstName"] = _pick_best(first_votes)
        merged["middleName"] = _pick_best(middle_votes)
        merged["lastName"] = _pick_best(last_votes)

        if gender_votes:
            merged["gender"] = sorted(gender_votes.keys(), key=lambda x: gender_votes[x], reverse=True)[0]

        if merged["fullName"] and merged["fullName"] in merged["nameAliases"]:
            merged["nameAliases"].remove(merged["fullName"])

        merged["nameAliases"] = [{"fullName": n} for n in sorted(merged["nameAliases"]) if n]
        merged["aliases"] = all_struct_aliases
        merged["addresses"] = all_addresses
        merged["datesOfBirthParsed"] = all_dob_parsed

        merged["datesOfBirthIso"] = sorted([x for x in merged["datesOfBirthIso"] if x])
        merged["datesOfDeathIso"] = sorted([x for x in merged["datesOfDeathIso"] if x])

        merged["nationalitiesIsoCodes"] = sorted(
            [x for x in merged["nationalitiesIsoCodes"] if isinstance(x, str) and x]
        )
        merged["profileImages"] = sorted([x for x in merged["profileImages"] if x])

        merged["nationality"] = list(merged["nationalitiesIsoCodes"])

        dob_seed = merged["datesOfBirthIso"][0] if merged["datesOfBirthIso"] else "NODOB"
        id_seed = f"{merged['fullName']}:{dob_seed}".strip()
        res_id = hashlib.sha256(id_seed.encode("utf-8")).hexdigest()

        merged["resourceId"] = res_id
        merged["resourceUri"] = f"/individuals/{res_id}"
        merged["qrCode"] = res_id[:10]

        merged["provider"] = self.provider
        merged["provider_version"] = int(time.time() * 1000)

        conflicts: Dict[str, Any] = {}
        if len(gender_values) > 1:
            conflicts["gender_values"] = sorted([str(x) for x in gender_values])
        if len(dob_year_values) > 1:
            conflicts["dob_years"] = sorted(dob_year_values)

        out_obj: Dict[str, Any] = {
            "data": merged,
            "datasets": sorted([x for x in final_datasets_list if isinstance(x, str) and x]),
        }
        if conflicts:
            out_obj["debug_trace"] = {"conflicts": conflicts}

        return out_obj

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
    def _save_final_output(self, merged_results: List[Dict[str, Any]]) -> None:
        """
        Saves the resolved Golden Records to disk.
        1. Loads previous hash state (if available).
        2. Calculates robust content hashes for all new profiles.
        3. Identifies DELTAS (changes vs previous state).
        4. Writes FULL export (all records).
        5. Writes DELTA export (changed records only).
        6. Persists new hash state for the next run.
        7. Generates Provider Metadata.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # 1. Setup Files & Load State
        # ---------------------------------------------------------
        full_out_filename = self.output_file
        if not full_out_filename.lower().endswith(".jsonl"):
            full_out_filename = full_out_filename.rsplit(".", 1)[0] + ".jsonl"
            
        full_gz_path = self.output_dir / (full_out_filename + ".gz")
        delta_gz_path = self.output_dir / "Delta_Export.jsonl.gz"

        old_hashes: Dict[str, str] = {}
        if self.state_file.exists():
            try:
                logger.info("Loading previous resolution state from: %s", self.state_file)
                with self.state_file.open("r", encoding="utf-8") as f:
                    old_hashes = json.load(f)
            except Exception as e:
                logger.warning("Failed to load state file (forcing full delta): %s", e)

        # ---------------------------------------------------------
        # 2. Process Records (Hash & Identify Deltas)
        # ---------------------------------------------------------
        delta_records: List[Dict[str, Any]] = []
        new_hashes: Dict[str, str] = {}
        all_datasets: Set[str] = set()
        total_records = 0

        # Helper to handle sets in JSON serialization
        def set_default(o):
            if isinstance(o, set):
                return list(o)
            return str(o)

        logger.info("Writing Full Export and calculating Deltas...")

        # We write the Full Export stream immediately to save memory
        with gzip.open(full_gz_path, "wt", encoding="utf-8") as f_full:
            for item in merged_results:
                if not isinstance(item, dict):
                    continue
                
                # A. Inject Robust Hash 
                curr_hash = self._compute_profile_hash(item)

                data = item.get("data", {})
                rid = data.get("resourceId")

                # Track datasets for metadata
                ds = item.get("datasets")
                if isinstance(ds, list):
                    all_datasets.update([x for x in ds if isinstance(x, str)])

                # B. Delta Detection
                if rid and curr_hash:
                    # Save this hash to the new state map
                    new_hashes[rid] = curr_hash

                    # Check if Changed (Upsert) or New (Insert)
                    # If rid wasn't in old_hashes, it's new.
                    # If rid is in old_hashes but hash differs, it's an update.
                    if rid not in old_hashes or old_hashes[rid] != curr_hash:
                        delta_records.append(item)

                # C. Write to Full Export
                f_full.write(json.dumps(item, ensure_ascii=False, default=set_default) + "\n")
                total_records += 1

        logger.info("Saved FULL Export (%d records) to: %s", total_records, full_gz_path)

        # ---------------------------------------------------------
        # 3. Write Delta Export
        # ---------------------------------------------------------
        with gzip.open(delta_gz_path, "wt", encoding="utf-8") as f_delta:
            for item in delta_records:
                f_delta.write(json.dumps(item, ensure_ascii=False, default=set_default) + "\n")

        logger.info("Saved DELTA Export (%d records) to: %s", len(delta_records), delta_gz_path)

        # ---------------------------------------------------------
        # 4. Persist New State
        # ---------------------------------------------------------
        try:
            with self.state_file.open("w", encoding="utf-8") as f_state:
                json.dump(new_hashes, f_state)
            logger.info("Persisted resolution state hashes to: %s", self.state_file)
        except Exception as e:
            logger.error("Failed to save resolution state: %s", e)

        # ---------------------------------------------------------
        # 5. Generate Provider Metadata
        # ---------------------------------------------------------
        # Calculate SHA256 of the FULL .jsonl.gz file for integrity verification
        sha256 = hashlib.sha256()
        with full_gz_path.open("rb") as rf:
            for chunk in iter(lambda: rf.read(1024 * 1024), b""):
                sha256.update(chunk)
        gz_sha256 = sha256.hexdigest()

        provider_body_path = self.output_dir / f"{self.provider}_meta.json"
        
        provider_body = {
            "sha256": gz_sha256,
            "file": str(full_gz_path.name),
            "format": "jsonl.gz",
            "records": total_records,
            "exported_at_iso": datetime.now(timezone.utc).isoformat(),
            "datasets": sorted(all_datasets),
            "source_dir": str(self.processed_dir),
            "max_block_size": self.max_block_size,
            # Delta metadata used by Provider to decide whether to run 'insert' or 'update'
            "delta": {
                "file": str(delta_gz_path.name),
                "records": len(delta_records),
                "strategy": "hash_diff"
            }
        }

        provider_body_path.write_text(
            json.dumps(provider_body, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
        logger.info("Saved provider metadata to: %s", provider_body_path)