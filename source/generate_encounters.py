import pandas as pd
import numpy as np
import math
import re
from pathlib import Path
import os
import json

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    _keep = set("abcdefghijklmnopqrstuvwxyz0123456789_/")
    def _norm(s):
        s = str(s).lower().replace("-", " ").replace(" ", "_")
        # Handle special Nidoran characters
        s = s.replace("♀", "_f").replace("♂", "_m")
        return "".join(ch for ch in s if ch in _keep).replace("__", "_").rstrip("_")
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        df[col] = df[col].map(_norm)
    return df

def load_trainer_data(gen):

    def load_trainer_data_gen1():
        path = Path(__file__).parent.parent / 'data/gen_1/TrainerDataGen1Raw.txt'
        null_location_mapping = {
            "Green1": "pallet_town", "Brock": "pewter_gym", "Misty": "cerulean_gym",
            "LtSurge": "vermilion_gym", "Erika": "celadon_gym", "Sabrina": "saffron_gym",
            "Koga": "fuchsia_gym", "Blaine": "cinnabar_gym", "Lorelei": "indigo_plateau",
            "Bruno": "indigo_plateau", "Agatha": "indigo_plateau", "Lance": "indigo_plateau",
            "Green3": "indigo_plateau"
        }
        out = []
        trainer = loc = None
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.endswith("Data:"):
                    trainer = line[:-5] if line.endswith("Data:") else line.rstrip(":")
                    loc = None
                    continue
                if line.startswith(";"):
                    loc = line[1:].strip()
                    continue
                if line.startswith("db"):
                    toks = [t.strip() for t in line[2:].strip().split(",")]
                    if toks and toks[0].startswith("$FF"):
                        seq = [t for t in toks if t not in {"0", "$FF"}]
                        for i in range(0, len(seq), 2):
                            lvl = int(seq[i]); mon = seq[i + 1]
                            locs = [loc] if loc else [None]
                            if loc and "/" in loc:
                                locs = [p.strip() for p in loc.split("/")]
                            for L in locs:
                                out.append((trainer, L, mon, lvl))
                    else:
                        lvl = int(toks[0])
                        mons = [m for m in toks[1:] if m != "0"]
                        locs = [loc] if loc else [None]
                        if loc and "/" in loc:
                            locs = [p.strip() for p in loc.split("/")]
                        for mon in mons:
                            for L in locs:
                                out.append((trainer, L, mon, lvl))
        df = pd.DataFrame(out, columns=["trainer_name_enc", "location_enc", "pokemon_enc", "level_enc"])
        df["trainer_name_enc"] = df["trainer_name_enc"].str.replace(r"Data$", "", regex=True)
        df["location_enc"] = df["location_enc"].fillna(df["trainer_name_enc"].map(null_location_mapping))
        df = normalize_text_columns(df)
        return df

    def load_trainer_data_gen2():
        path = Path(__file__).parent.parent / 'data/gen_2/TrainerDataGen2Raw.txt'
        COLS = ["trainer_name_enc", "location_enc", "pokemon_enc", "level_enc",
                "hp_enc", "attack_enc", "defense_enc", "sp_attack_enc", "sp_defense_enc", "speed_enc",
                "types_enc", "move1_name_enc", "move2_name_enc", "move3_name_enc", "move4_name_enc"]
        STAT_KEYS = ["Max HP", "Attack", "Defense", "Sp. Atk.", "Sp. Def.", "Speed"]
        MOVE_KEYS = ["Move 1", "Move 2", "Move 3", "Move 4"]
        ALL_KEYS = ["Level", "Type", "Item"] + MOVE_KEYS + STAT_KEYS

        def _clean(v):
            if v is None or v == "-" or v == "":
                return None
            m = re.search(r"\d+", str(v))
            return int(m.group()) if m else str(v).strip()

        with open(path, encoding="utf-8") as f:
            text = f.read()

        blocks = re.split(r"\n\.\s*\n", text.strip())
        out = []
        for block in blocks:
            lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
            if not lines: continue
            head = lines[0]
            parts = [p.strip() for p in head.split(" - ")]
            trainer = parts[-1]
            location = " - ".join(parts[:-1]) if len(parts) > 1 else None
            idx_np = next((i for i, l in enumerate(lines) if l.startswith("Number of Pokemon:")), None)
            if idx_np is None: continue
            declared_n = int(re.search(r"\d+", lines[idx_np]).group())
            names, i = [], idx_np + 1
            while i < len(lines) and not any(lines[i].startswith(k) for k in ALL_KEYS):
                vals = re.split(r"\t+|\s{2,}", lines[i].strip())
                names.extend([v for v in vals if v])
                i += 1
            table = {k: [] for k in ALL_KEYS}
            while i < len(lines):
                ln = lines[i]
                if ln.startswith("EXP. Yield") or ln.startswith("Note on EXP") or ln.startswith("Trainer Stat Exp. Yields"):
                    break
                for k in ALL_KEYS:
                    if ln.startswith(k):
                        vals = re.split(r"\t+", ln)[1:]
                        if not vals:
                            vals = re.split(r"\s{2,}", ln.replace(k, "", 1).strip())
                        if k == "Level":
                            vals = [re.sub(r"Level\s*", "", v) for v in vals]
                        table[k] = (table[k] + vals)
                        break
                i += 1
            true_n = max(declared_n, len(names), max((len(v) for v in table.values()), default=0))
            names += [None] * (true_n - len(names))
            for k in ALL_KEYS:
                table[k] += [None] * (true_n - len(table[k]))
            for j in range(true_n):
                out.append({
                    "trainer_name_enc": trainer,
                    "location_enc": location,
                    "pokemon_enc": names[j],
                    "level_enc": _clean(table["Level"][j]),
                    "hp_enc": _clean(table["Max HP"][j]),
                    "attack_enc": _clean(table["Attack"][j]),
                    "defense_enc": _clean(table["Defense"][j]),
                    "sp_attack_enc": _clean(table["Sp. Atk."][j]),
                    "sp_defense_enc": _clean(table["Sp. Def."][j]),
                    "speed_enc": _clean(table["Speed"][j]),
                    "types_enc": table["Type"][j].strip() if table["Type"][j] else None,
                    "move1_name_enc": table["Move 1"][j],
                    "move2_name_enc": table["Move 2"][j],
                    "move3_name_enc": table["Move 3"][j],
                    "move4_name_enc": table["Move 4"][j],
                })
        df = pd.DataFrame(out, columns=COLS)
        df = normalize_text_columns(df)
        return df

    def load_trainer_data_gen3():
        path = Path(__file__).parent.parent / 'data/gen_3/TrainerDataGen3Raw.csv'
        df = pd.read_csv(path)
        column_map = {
            'Trainer': 'trainer_name_enc',
            'Route': 'location_enc',
            'Location': 'location_enc_detail',
            'Pokemon': 'pokemon_enc',
            'Level': 'level_enc',
            'Attack 1': 'move1_name_enc',
            'Attack 2': 'move2_name_enc',
            'Attack 3': 'move3_name_enc',
            'Attack 4': 'move4_name_enc',
            'HP': 'hp_enc',
            'Attack': 'attack_enc',
            'Defense': 'defense_enc',
            'Sp. Attack': 'sp_attack_enc',
            'Sp Defense': 'sp_defense_enc',
            'Speed': 'speed_enc'
        }
        df = df.rename(columns=column_map)
        df = normalize_text_columns(df)
        return df

    if gen == 1:
        return load_trainer_data_gen1()
    elif gen == 2:
        return load_trainer_data_gen2()
    elif gen == 3:
        return load_trainer_data_gen3()
    else:
        return None

def apply_location_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply location overrides from TrainerDataRawAdjustments.csv.
    
    Args:
        df (pd.DataFrame): Trainer data with location_enc column
        
    Returns:
        pd.DataFrame: Data with location_enc values overridden where matches exist
    """
    adjustments_path = Path(__file__).parent.parent / 'data/gen_all/TrainerDataRawAdjustments.csv'
    
    # Check if adjustments file exists and has content
    if not adjustments_path.exists():
        print("TrainerDataRawAdjustments.csv not found - skipping location adjustments")
        return df
    
    try:
        adjustments = pd.read_csv(adjustments_path, encoding='utf-8-sig')
        if adjustments.empty:
            print("TrainerDataRawAdjustments.csv is empty - skipping location adjustments")
            return df
            
        # Verify required columns exist
        required_cols = ['trainer_name_enc', 'location_enc', 'new_location_enc']
        if not all(col in adjustments.columns for col in required_cols):
            print(f"TrainerDataRawAdjustments.csv missing required columns: {required_cols}")
            return df
        
        # Create adjustment lookup lists (exact matches and wildcard matches)
        # Process BEFORE normalization to properly detect blank values
        exact_adjustments = {}
        wildcard_adjustments = []
        
        for _, row in adjustments.iterrows():
            # Check for blank values BEFORE normalization
            trainer_raw = row['trainer_name_enc']
            location_raw = row['location_enc'] 
            
            # Check if trainer field is effectively blank (NaN, empty, or whitespace only)
            trainer = None
            if pd.notna(trainer_raw):
                trainer_str = str(trainer_raw).strip()
                if trainer_str:  # Not empty after stripping
                    trainer = trainer_str
            
            # Check if location field is effectively blank 
            location = None
            if pd.notna(location_raw):
                location_str = str(location_raw).strip() 
                if location_str:  # Not empty after stripping
                    location = location_str
                    
            new_location = str(row['new_location_enc']).strip()
            
            if trainer is not None and location is not None:
                # Exact match - both trainer and location specified
                # Apply normalization to the key values for matching
                _keep = set("abcdefghijklmnopqrstuvwxyz0123456789_/")
                def _norm(s):
                    s = str(s).lower().replace("-", " ").replace(" ", "_")
                    return "".join(ch for ch in s if ch in _keep).replace("__", "_").rstrip("_")
                
                key = (_norm(trainer), _norm(location))
                exact_adjustments[key] = _norm(new_location)
            else:
                # Wildcard match - at least one field is blank
                # Apply normalization to non-None values
                _keep = set("abcdefghijklmnopqrstuvwxyz0123456789_/")
                def _norm(s):
                    s = str(s).lower().replace("-", " ").replace(" ", "_")
                    return "".join(ch for ch in s if ch in _keep).replace("__", "_").rstrip("_")
                
                norm_trainer = _norm(trainer) if trainer is not None else None
                norm_location = _norm(location) if location is not None else None
                norm_new_location = _norm(new_location)
                wildcard_adjustments.append((norm_trainer, norm_location, norm_new_location))
        
        if not exact_adjustments and not wildcard_adjustments:
            print("No valid adjustments found - skipping location adjustments")
            return df
        
        # Apply adjustments
        df_result = df.copy()
        adjustments_applied = 0
        
        # Normalize the encounter data for matching
        df_normalized = normalize_text_columns(df_result)
        
        for idx, row in df_normalized.iterrows():
            trainer_name = row['trainer_name_enc']
            location_name = row['location_enc']
            new_location = None
            
            # First check for exact matches
            exact_key = (trainer_name, location_name)
            if exact_key in exact_adjustments:
                new_location = exact_adjustments[exact_key]
            else:
                # Check wildcard matches
                for wildcard_trainer, wildcard_location, wildcard_new_location in wildcard_adjustments:
                    trainer_match = (wildcard_trainer is None) or (wildcard_trainer == trainer_name)
                    location_match = (wildcard_location is None) or (wildcard_location == location_name)
                    
                    if trainer_match and location_match:
                        new_location = wildcard_new_location
                        break  # Use first matching wildcard rule
            
            if new_location:
                # Apply the change to the original (non-normalized) dataframe
                df_result.at[idx, 'location_enc'] = new_location
                adjustments_applied += 1
        
        print(f"Applied {adjustments_applied} location adjustments from TrainerDataRawAdjustments.csv")
        return df_result
        
    except Exception as e:
        print(f"Error processing TrainerDataRawAdjustments.csv: {e}")
        return df


def add_stages(gen, df):
    if gen == 1:
        stages_csv = Path(__file__).parent.parent / 'data/gen_1/stages_gen_1.csv'
    elif gen == 2:
        stages_csv = Path(__file__).parent.parent / 'data/gen_2/stages_gen_2.csv'
    elif gen == 3:
        stages_csv = Path(__file__).parent.parent / 'data/gen_3/stages_gen_3.csv'
    else:
        return df
    stages = pd.read_csv(stages_csv)
    stages = normalize_text_columns(stages)
    loc_list = stages["location"].tolist()
    loc2stage = dict(zip(stages["location"], stages["location_stage"]))
    def match_stage(loc):
        if pd.isna(loc):
            return pd.NA
        hits = [l for l in loc_list if l in loc]
        if not hits:
            return pd.NA
        return loc2stage[max(hits, key=len)]
    out = df.copy()
    out["stage_enc"] = out["location_enc"].apply(match_stage).astype("Int64")
    return out

def calc_enc_stats(gen, df):
    def gen1(df):
        stats_csv = Path(__file__).parent.parent / 'data/gen_2/stats_gen_2.csv'
        ivs = {"hp": 8, "attack": 9, "defense": 8, "sp_attack": 8, "sp_defense": 8, "speed": 8}
        ev = 0
        stats = pd.read_csv(stats_csv)
        stats = normalize_text_columns(stats)
        base = stats.set_index("pokemon")[["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "types"]]
        out = df.copy()
        out = out.join(base, on="pokemon_enc", how="left", rsuffix="_base")
        L = out["level_enc"].astype(int)
        EV = ev
        ev_term = math.floor(math.ceil(math.sqrt(EV)) / 4)
        def calc(B, I, hp=False):
            return np.floor((((B + I) * 2 + ev_term) * L) / 100) + (L + 10 if hp else 5)
        out["hp_enc"] = calc(out["hp"], ivs.get("hp", 8), hp=True).astype(int)
        out["attack_enc"] = calc(out["attack"], ivs.get("attack", 8)).astype(int)
        out["defense_enc"] = calc(out["defense"], ivs.get("defense", 8)).astype(int)
        
        # Gen 1 has only one "Special" stat - use sp_attack from Gen 2 data as the Special stat
        # Both sp_attack_enc and sp_defense_enc should be identical in Gen 1
        special_stat = calc(out["sp_attack"], ivs.get("sp_attack", 8)).astype(int)
        out["sp_attack_enc"] = special_stat
        out["sp_defense_enc"] = special_stat
        
        out["speed_enc"] = calc(out["speed"], ivs.get("speed", 8)).astype(int)
        out["types_enc"] = out["types"]
        out = out.drop(columns=["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "types"])
        return out

    def gen2(df):
        """Handle Gen 2 - calculate stats using DV values from trainer_dvs.csv"""
        
        def create_trainer_name_mapping():
            """Create mapping from trainer_dvs.csv format to normalized trainer names"""
            mapping = {
                # Gym Leaders - these follow pattern 'Gym Leader X' -> 'gym_leader_x'
                "Lt. Surge": "gym_leader_lt_surge",
                "Falkner": "gym_leader_falkner", 
                "Bugsy": "gym_leader_bugsy",
                "Whitney": "gym_leader_whitney",
                "Morty": "gym_leader_morty",
                "Chuck": "gym_leader_chuck", 
                "Jasmine": "gym_leader_jasmine",
                "Pryce": "gym_leader_pryce",
                "Clair": "gym_leader_clair",
                
                # Elite 4 and Champion - these follow pattern 'Elite Four X' -> 'elite_four_x'
                "Will": "elite_four_will",
                "Bruno": "elite_four_bruno", 
                "Karen": "elite_four_karen",
                "Koga": "elite_four_koga",
                "Lance": "champion_lance",
                
                # Kanto Gym Leaders - pattern varies, need to check encounters
                "Brock": "gym_leader_brock",
                "Misty": "gym_leader_misty",
                "Erika": "gym_leader_erika", 
                "Janine": "gym_leader_janine",
                "Sabrina": "gym_leader_sabrina",
                "Blaine": "gym_leader_blaine",
                "Blue": "gym_leader_blue",  # or champion_blue
                
                # Special trainers
                "Silver": "rival_silver",
                "Red": "pokemon_trainer_red",  # Mt. Silver
                "EusineC": "eusine",  # Might need adjustment
                "Cal": "cal",  # Need to check actual name in encounters
                
                # Regular trainer classes - these should match directly after normalization
                "Scientist": "scientist",
                "Youngster": "youngster", 
                "Schoolboy": "schoolboy",
                "Bird Keeper": "bird_keeper",
                "Lass": "lass",
                "Cooltrainer": "cooltrainer",
                "Beauty": "beauty",
                "Pokémaniac": "pokemaniac",
                "Rocket Grunt": "rocket_grunt",
                "Rocket Executive": "rocket_executive",
                "Gentleman": "gentleman",
                "Skier": "skier", 
                "Teacher": "teacher",
                "Bug Catcher": "bug_catcher",
                "Fisherman": "fisherman",
                "Swimmer": "swimmer",
                "Sailor": "sailor",
                "Super Nerd": "super_nerd",
                "Guitarist": "guitarist",
                "Hiker": "hiker",
                "Biker": "biker",
                "Burglar": "burglar",
                "Firebreather": "firebreather",
                "Juggler": "juggler", 
                "Blackbelt": "blackbelt",
                "Psychic": "psychic",
                "Picnicker": "picnicker",
                "Camper": "camper",
                "Sage": "sage",
                "Medium": "medium",
                "Boarder": "boarder",
                "Pokéfan": "pokefan",
                "Kimono Girl": "kimono_girl",
                "Twins": "twins",
                "Officer": "officer"
            }
            return mapping
        
        def normalize_trainer_name(name):
            """Apply same normalization as used elsewhere in the codebase"""
            _keep = set("abcdefghijklmnopqrstuvwxyz0123456789_/")
            s = str(name).lower().replace("-", " ").replace(" ", "_")
            # Handle special Nidoran characters
            s = s.replace("♀", "_f").replace("♂", "_m")
            return "".join(ch for ch in s if ch in _keep).replace("__", "_").rstrip("_")
        
        # Load stats and DV data
        stats_csv = Path(__file__).parent.parent / 'data/gen_2/stats_gen_2.csv'
        dvs_csv = Path(__file__).parent.parent / 'data/gen_2/trainer_dvs.csv'
        
        stats = pd.read_csv(stats_csv)
        dvs_df = pd.read_csv(dvs_csv)
        
        stats = normalize_text_columns(stats)
        base = stats.set_index("pokemon")[["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "types"]]
        
        # Create trainer name mapping and DV lookup
        trainer_mapping = create_trainer_name_mapping()
        
        # Build DV lookup - map from normalized trainer name to DV values
        dv_lookup = {}
        silver_late_dvs = None
        
        for _, row in dvs_df.iterrows():
            trainer_name = row['Trainer']
            
            # Handle special case for Silver Late
            if trainer_name == 'Silver Late':
                silver_late_dvs = {
                    'attack': int(row['Attack DV']),
                    'defense': int(row['Defense DV']), 
                    'speed': int(row['Speed DV']),
                    'special': int(row['Special DV'])
                }
                continue  # Don't add to main lookup, handled specially
            
            # First try direct mapping
            mapped_name = trainer_mapping.get(trainer_name)
            if mapped_name:
                normalized_name = mapped_name
            else:
                # Fallback to normalization
                normalized_name = normalize_trainer_name(trainer_name)
            
            dv_lookup[normalized_name] = {
                'attack': int(row['Attack DV']),
                'defense': int(row['Defense DV']), 
                'speed': int(row['Speed DV']),
                'special': int(row['Special DV'])  # Used for both sp_attack and sp_defense in Gen 2
            }
        
        # Default DVs if trainer not found
        default_dvs = {'attack': 9, 'defense': 8, 'speed': 8, 'special': 8}
        
        # Debug: Print mapping summary
        print(f"  - Built DV lookup for {len(dv_lookup)} trainers from trainer_dvs.csv")
        if silver_late_dvs:
            print(f"  - Silver Late DVs loaded for victory_road/mt_moon: {silver_late_dvs}")
        mapped_trainers = [name for name in dv_lookup.keys()][:10]  # Show first 10
        print(f"  - Sample mapped trainers: {mapped_trainers}")
        
        out = df.copy()
        out = out.join(base, on="pokemon_enc", how="left", rsuffix="_base")
        
        # Debug: Check for any NA values in level_enc
        na_levels = out["level_enc"].isna().sum()
        if na_levels > 0:
            print(f"Warning: Found {na_levels} NA values in level_enc, dropping these rows")
            out = out.dropna(subset=["level_enc"])
        
        ev = 0
        ev_term = math.floor(math.ceil(math.sqrt(ev)) / 4)
        
        def calc(B, I, L, hp=False):
            return np.floor((((B + I) * 2 + ev_term) * L) / 100) + (L + 10 if hp else 5)
        
        # Calculate stats for each row using trainer-specific DVs
        for idx, row in out.iterrows():
            trainer_name = row['trainer_name_enc']
            location_name = row['location_enc']
            level = int(row['level_enc'])
            
            # Get DV values for this trainer, with special handling for Silver Late
            if trainer_name == 'rival_silver' and location_name in ['victory_road', 'mt_moon']:
                # Use Silver Late DVs for these specific locations
                trainer_dvs = silver_late_dvs if silver_late_dvs else default_dvs
            else:
                # Use normal DV lookup
                trainer_dvs = dv_lookup.get(trainer_name, default_dvs)
            
            # Calculate stats using trainer's specific DVs
            # For HP, we need to use a different DV calculation in Gen 2
            # HP DV = (Attack DV % 2) * 8 + (Defense DV % 2) * 4 + (Speed DV % 2) * 2 + (Special DV % 2)
            hp_dv = ((trainer_dvs['attack'] % 2) * 8 + 
                     (trainer_dvs['defense'] % 2) * 4 + 
                     (trainer_dvs['speed'] % 2) * 2 + 
                     (trainer_dvs['special'] % 2))
            
            out.at[idx, "hp_enc"] = int(calc(row["hp"], hp_dv, level, hp=True))
            out.at[idx, "attack_enc"] = int(calc(row["attack"], trainer_dvs['attack'], level))
            out.at[idx, "defense_enc"] = int(calc(row["defense"], trainer_dvs['defense'], level))
            out.at[idx, "sp_attack_enc"] = int(calc(row["sp_attack"], trainer_dvs['special'], level))
            out.at[idx, "sp_defense_enc"] = int(calc(row["sp_defense"], trainer_dvs['special'], level))
            out.at[idx, "speed_enc"] = int(calc(row["speed"], trainer_dvs['speed'], level))
        
        out["types_enc"] = out["types"]
        out = out.drop(columns=["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "types"])
        return out
    
    def gen3(df):
        """Handle Gen 3 - add types_enc from stats file but preserve existing stats"""
        stats_csv = Path(__file__).parent.parent / f'data/gen_3/stats_gen_3.csv'
        stats = pd.read_csv(stats_csv)
        stats = normalize_text_columns(stats)
        
        # Gen 3 uses abbreviated column names, but we only need types
        base = stats.set_index("pokemon")[["types"]]
        
        out = df.copy()
        # Join only the types column, preserving existing stats from the raw data
        out = out.join(base, on="pokemon_enc", how="left", rsuffix="_base")
        out["types_enc"] = out["types"]
        
        # Clean up temporary column
        if "types" in out.columns:
            out = out.drop(columns=["types"])
        
        return out

    if gen == 1:
        return gen1(df)
    elif gen == 2:
        return gen2(df)
    elif gen == 3:
        return gen3(df)
    else:
        return df


def calc_enc_moves(gen, df=None):
    high_crit_raw = [
        "Aeroblast", "Air Cutter", "Blaze Kick", "Crabhammer", "Cross Chop",
        "Karate Chop", "Leaf Blade", "Poison Tail", "Razor Leaf", "Razor Wind",
        "Sky Attack", "Slash"
    ]
    physical_types = {"normal", "fighting", "poison", "ground", "flying", "bug", "rock", "ghost", "steel"}
    special_types = {"fire", "water", "electric", "grass", "ice", "psychic", "dragon", "dark"}
    def classify_type(t):
        if not isinstance(t, str):
            return "ERROR"
        t = t.lower().replace(" ", "_")
        if t in physical_types:
            return "physical"
        if t in special_types:
            return "special"
        return "ERROR"

    normalized_crit_set = {s.lower().replace(" ", "_") for s in high_crit_raw}

    if gen == 1:
        # parse learnsets
        def _parse_learnsets_gen1(path):
            rows = []
            poke = mode = None
            ready = False
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            end = re.compile(r'^(Level-up learnset|TM/HM learnset|Pre-evolution|GSC|Event|Illegal|#\d{3}:|Type)')
            for ln in map(str.rstrip, lines):
                m = re.match(r'^#\d{3}:\s*(.+)$', ln)
                if m:
                    poke, mode, ready = m.group(1).strip(), None, False
                    continue
                if ln.startswith("Level-up learnset"):
                    mode, ready = "lvl", False
                    continue
                if ln.startswith("TM/HM learnset"):
                    mode, ready = "tm", False
                    continue
                if mode and end.match(ln):
                    mode, ready = None, False
                if not mode:
                    continue
                if not ready:
                    if ln.strip() in ("RBY", "RB", "RB Y", "RB\tY"):
                        ready = True
                    continue
                if not ln.strip():
                    continue
                if mode == "lvl":
                    m = re.match(r'\s*(\d+)\s+(.*\S)', ln)
                    if m:
                        lvl, rest = int(m.group(1)), m.group(2).strip()
                        toks = rest.split()
                        if toks and toks[0].isdigit():
                            rest = " ".join(toks[1:])
                        rows.append((poke, rest, lvl))
                else:
                    if re.search(r'(TM|HM)\d{2}', ln):
                        mv = ln.split()[-1]
                        rows.append((poke, mv, 0))
            return pd.DataFrame(rows, columns=["pokemon", "move_name", "move_level"])

        learnset_path = Path(__file__).parent.parent / 'data/gen_1/pokemon_data_alt.txt'
        df_learn = _parse_learnsets_gen1(learnset_path)
        df_learn = normalize_text_columns(df_learn)

        meta_path = Path(__file__).parent.parent / f'data/gen_1/unique_moves_alt_gen_1.csv'
        df_meta = pd.read_csv(meta_path)
        df_meta = df_meta.rename(columns={
            "Move": "move_name",
            "Type": "move_type",
            "Power": "move_power",
            "Acc": "move_accuracy"
        })
        df_meta["move_category"] = pd.NA  # placeholder; will be set below
        df_meta = normalize_text_columns(df_meta)
        for col in ("move_power", "move_accuracy"):
            if col in df_meta.columns:
                df_meta[col] = pd.to_numeric(df_meta[col], errors="coerce")

        # assign category based on type
        df_meta["move_category"] = df_meta["move_type"].apply(classify_type)

        out = df_learn.merge(df_meta, on="move_name", how="left")
        out["move_crit"] = out["move_name"].isin(normalized_crit_set)
        return out[["pokemon", "move_name", "move_level",
                    "move_type", "move_category", "move_power",
                    "move_accuracy", "move_crit"]]
    else:
        path = Path(__file__).parent.parent / f'data/gen_{gen}/unique_moves_alt_gen_{gen}.csv'
        df_moves = pd.read_csv(path)
        rename_map = {
            "move": "move_name", "Move": "move_name",
            "type": "move_type", "Type": "move_type",
            "power": "move_power", "Power": "move_power",
            "accuracy": "move_accuracy", "Accuracy": "move_accuracy",
            "crit": "move_crit", "Crit": "move_crit",
        }
        df_moves = df_moves.rename(columns=rename_map)
        df_moves["move_category"] = df_moves.get("move_type", pd.NA).apply(classify_type)
        df_moves = normalize_text_columns(df_moves)
        for col in ("move_power", "move_accuracy"):
            if col in df_moves.columns:
                df_moves[col] = pd.to_numeric(df_moves[col], errors="coerce")
        df_moves["move_crit"] = df_moves["move_name"].isin(normalized_crit_set)
        expected = ["move_name", "move_type", "move_category", "move_power", "move_accuracy", "move_crit"]
        if "move_level" in df_moves.columns:
            expected.append("move_level")
        cols = [c for c in expected if c in df_moves.columns] + [c for c in df_moves.columns if c not in expected]
        return df_moves[cols]


def add_encounter_moves(gen, enc_df: pd.DataFrame, moves_df: pd.DataFrame) -> pd.DataFrame:
    high_crit_raw = [
        "Aeroblast","Air Cutter","Blaze Kick","Crabhammer","Cross Chop",
        "Karate Chop","Leaf Blade","Poison Tail","Razor Leaf","Razor Wind",
        "Sky Attack","Slash"
    ]
    physical_types = {"normal", "fighting", "poison", "ground", "flying", "bug", "rock", "ghost", "steel"}
    special_types = {"fire", "water", "electric", "grass", "ice", "psychic", "dragon", "dark"}
    def classify_type(t):
        if not isinstance(t, str):
            return "ERROR"
        t = t.lower().replace(" ", "_")
        if t in physical_types:
            return "physical"
        if t in special_types:
            return "special"
        return "ERROR"

    def _norm(s):
        s = str(s).lower().replace(" ", "_")
        # Handle special Nidoran characters
        s = s.replace("♀", "_f").replace("♂", "_m")
        return "".join(ch for ch in s if ch in set("abcdefghijklmnopqrstuvwxyz0123456789_/")).replace("__","_").rstrip("_")
    def normalize(df):
        df = df.copy()
        for c in df.select_dtypes(include=["object","string","category"]).columns:
            df[c] = df[c].map(_norm)
        return df

    alt_meta_path = Path(__file__).parent.parent / f"data/gen_{gen}/unique_moves_alt_gen_{gen}.csv"
    alt_meta = pd.read_csv(alt_meta_path).rename(columns={
        "Move":"move_name","Type":"move_type","Power":"move_power","Acc":"move_accuracy"
    })
    if "move_category" not in alt_meta.columns:
        alt_meta["move_category"] = alt_meta["move_type"].apply(lambda x: classify_type(str(x)))
    alt_meta = normalize(alt_meta)
    for c in ["move_power","move_accuracy"]:
        alt_meta[c] = pd.to_numeric(alt_meta[c], errors="coerce")
    alt_meta_dict = alt_meta.set_index("move_name")[["move_type","move_category","move_power","move_accuracy"]].to_dict("index")

    high_crit = { _norm(m) for m in high_crit_raw }

    enc_df = normalize(enc_df)
    moves_df = normalize(moves_df)

    if "move" in moves_df.columns:
        moves_df = moves_df.rename(columns={"move":"move_name","type":"move_type","power":"move_power","accuracy":"move_accuracy"})

    def _merge_meta_row(r):
        d = {
            "move_type": r.get("move_type"),
            "move_category": r.get("move_category") if r.get("move_category") is not None else classify_type(r.get("move_type")),
            "move_power": r.get("move_power"),
            "move_accuracy": r.get("move_accuracy")
        }
        missing = lambda v: v is None or (isinstance(v, float) and np.isnan(v))
        if any(missing(d[k]) for k in d):
            alt = alt_meta_dict.get(r.get("move_name",""), {})
            for k in ["move_type","move_category","move_power","move_accuracy"]:
                if missing(d[k]):
                    d[k] = alt.get(k)
        return d

    if "move_name" in moves_df.columns:
        move_meta_dict = {}
        for _, r in moves_df.iterrows():
            name = r["move_name"]
            m = _merge_meta_row(r)
            if name not in move_meta_dict:
                move_meta_dict[name] = {
                    "move_type": m.get("move_type"),
                    "move_category": m.get("move_category"),
                    "move_power": m.get("move_power"),
                    "move_accuracy": m.get("move_accuracy")
                }
    else:
        move_meta_dict = alt_meta_dict

    can_sim = {"pokemon","move_name","move_level"}.issubset(moves_df.columns)

    def _fill_from_catalog(row, names):
        out = []
        for n in names:
            if n:
                meta = move_meta_dict.get(n, {})
                t = meta.get("move_type")
                cat = meta.get("move_category")
                pw = meta.get("move_power")
                ac = meta.get("move_accuracy")
                cr = (n in high_crit)
            else:
                t = cat = pw = ac = None
                cr = False
            out.append((n, t, cat, pw, ac, cr))
        while len(out) < 4:
            out.append((None, None, None, None, None, False))
        names_list = [o[0] for o in out]
        types = [o[1] for o in out]
        cats = [o[2] for o in out]
        power = [o[3] for o in out]
        acc = [o[4] for o in out]
        crit = [o[5] for o in out]
        return pd.Series(names_list + types + cats + power + acc + crit)

    def _simulate_from_learnset(row, grouped):
        seq = grouped.get(row.pokemon_enc, [])
        names, types, cats, powers, accs, crits = [], [], [], [], [], []
        for lv, n, cat in seq:
            if lv > row.level_enc or n in names:
                continue
            if len(names) == 4:
                names.pop(0); types.pop(0); cats.pop(0); powers.pop(0); accs.pop(0); crits.pop(0)
            meta = move_meta_dict.get(n, {})
            t = meta.get("move_type")
            c = meta.get("move_category") if meta.get("move_category") is not None else classify_type(t)
            pw = meta.get("move_power")
            ac = meta.get("move_accuracy")
            cr = (n in high_crit)
            names.append(n); types.append(t); cats.append(c); powers.append(pw); accs.append(ac); crits.append(cr)
        while len(names) < 4:
            names.append(None); types.append(None); cats.append(None); powers.append(None); accs.append(None); crits.append(False)
        return pd.Series(names + types + cats + powers + accs + crits)

    cols = (
        [f"move_name_{i}_enc" for i in range(1, 5)] +
        [f"move_type_{i}_enc" for i in range(1, 5)] +
        [f"move_category_{i}_enc" for i in range(1, 5)] +
        [f"move_power_{i}_enc" for i in range(1, 5)] +
        [f"move_accuracy_{i}_enc" for i in range(1, 5)] +
        [f"move_crit_{i}_enc" for i in range(1, 5)]
    )

    if can_sim:
        lvl_moves = moves_df[moves_df.move_level > 0].copy()
        lvl_moves = lvl_moves.sort_values(["pokemon", "move_level"])
        grouped = {}
        for p, g in lvl_moves.groupby("pokemon"):
            seq = []
            for _, r in g.iterrows():
                seq.append((r["move_level"], r["move_name"], r.get("move_category", None)))
            grouped[p] = seq
        enc_df[cols] = enc_df.apply(lambda r: _simulate_from_learnset(r, grouped), axis=1)
    else:
        name_cols = [f"move{i}_name_enc" for i in range(1, 5)]
        if not all(c in enc_df.columns for c in name_cols):
            enc_df[[*cols]] = None
            return enc_df
        enc_df[cols] = enc_df.apply(lambda r: _fill_from_catalog(r, [r[c] for c in name_cols]), axis=1)

    if gen == 1:
        alt_override = normalize_text_columns(pd.read_csv(
            Path(__file__).parent.parent / 'data/gen_1/alternate_moves_gen_1.csv'
        ))
        override_map = {
            (r.trainer_name_enc, r.location_enc, r.pokemon_enc):
            [m for m in [r.move1_name_enc, r.move2_name_enc, r.move3_name_enc, r.move4_name_enc] if pd.notna(m)]
            for _, r in alt_override.iterrows()
        }
        def apply_override(row):
            key = (row.trainer_name_enc, row.location_enc, row.pokemon_enc)
            if key not in override_map:
                return row
            mv_names = override_map[key] + [None] * 4
            for i in range(4):
                mn = mv_names[i]
                row[f"move_name_{i+1}_enc"] = mn
                if mn and mn in move_meta_dict:
                    meta = move_meta_dict[mn]
                    row[f"move_type_{i+1}_enc"] = meta.get("move_type")
                    row[f"move_category_{i+1}_enc"] = meta.get("move_category") or classify_type(meta.get("move_type"))
                    row[f"move_power_{i+1}_enc"] = meta.get("move_power")
                    row[f"move_accuracy_{i+1}_enc"] = meta.get("move_accuracy")
                    row[f"move_crit_{i+1}_enc"] = (mn in {m.lower().replace(" ", "_") for m in high_crit_raw})
                else:
                    row[f"move_type_{i+1}_enc"] = None
                    row[f"move_category_{i+1}_enc"] = None
                    row[f"move_power_{i+1}_enc"] = None
                    row[f"move_accuracy_{i+1}_enc"] = None
                    row[f"move_crit_{i+1}_enc"] = False
            return row
        enc_df = enc_df.apply(apply_override, axis=1)

    def finalize_encounter_moves(df: pd.DataFrame) -> pd.DataFrame:
        base_cols = [
            "trainer_name_enc", "location_enc", "pokemon_enc", "level_enc",
            "stage_enc", "hp_enc", "attack_enc", "defense_enc",
            "sp_attack_enc", "sp_defense_enc", "speed_enc", "types_enc"
        ]
        move_name_cols = [f"move_name_{i}_enc" for i in range(1, 5)]
        move_type_cols = [f"move_type_{i}_enc" for i in range(1, 5)]
        move_category_cols = [f"move_category_{i}_enc" for i in range(1, 5)]
        move_power_cols = [f"move_power_{i}_enc" for i in range(1, 5)]
        move_accuracy_cols = [f"move_accuracy_{i}_enc" for i in range(1, 5)]
        move_crit_cols = [f"move_crit_{i}_enc" for i in range(1, 5)]
        desired_order = (
            base_cols + move_name_cols + move_type_cols +
            move_category_cols + move_power_cols +
            move_accuracy_cols + move_crit_cols
        )
        legacy_move_name = [f"move{i}_name_enc" for i in range(1, 5)]
        df = df.drop(columns=[c for c in legacy_move_name if c in df.columns], errors="ignore")
        if "location_enc_detail" in df.columns:
            df = df.drop(columns=["location_enc_detail"], errors="ignore")
        for col in desired_order:
            if col not in df.columns:
                df[col] = pd.NA
        return df[desired_order]

    enc_df = finalize_encounter_moves(enc_df)
    return enc_df


def add_evo_info_enc(gen: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add evo_id_enc and evo_index_enc columns to encounters based on pokemon_enc.
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        df (pd.DataFrame): Encounters dataframe
        
    Returns:
        pd.DataFrame: Encounters dataframe with evo_id_enc and evo_index_enc columns
    """
    def _norm(s):
        s = str(s).lower().replace(" ", "_")
        # Handle special Nidoran characters
        s = s.replace("♀", "_f").replace("♂", "_m")
        return "".join(ch for ch in s if ch in set("abcdefghijklmnopqrstuvwxyz0123456789_/")).replace("__", "_").rstrip("_")
    
    # Load stats CSV for the generation
    stats_path = Path(__file__).parent.parent / f'data/gen_{gen}/stats_gen_{gen}.csv'
    try:
        stats_df = pd.read_csv(stats_path)
        
        # Normalize pokemon names for matching
        stats_df["pokemon_norm"] = stats_df["pokemon"].map(_norm)
        
        # Calculate evo_index for each pokemon within its evo_id group
        stats_df_sorted = stats_df.sort_values(['evo_id', 'evo_lvl']).reset_index(drop=True)
        stats_df_sorted['evo_index'] = stats_df_sorted.groupby('evo_id').cumcount() + 1
        
        # Create lookup dictionary for evo_id and evo_index
        evo_lookup = dict(zip(stats_df_sorted['pokemon_norm'], 
                             zip(stats_df_sorted['evo_id'], stats_df_sorted['evo_index'])))
        
        # Add evo_id_enc and evo_index_enc columns to encounters
        df = df.copy()
        
        def get_evo_info(pokemon_enc):
            pokemon_norm = _norm(pokemon_enc) if pd.notna(pokemon_enc) else ""
            return evo_lookup.get(pokemon_norm, (pd.NA, pd.NA))
        
        # Apply the lookup to get both evo_id and evo_index
        evo_info = df['pokemon_enc'].apply(get_evo_info)
        df['evo_id_enc'] = [info[0] for info in evo_info]
        df['evo_index_enc'] = [info[1] for info in evo_info]
        
        # Insert these columns right after enc_id (first column) to match player_pokemon structure
        cols = list(df.columns)
        # Remove the newly added columns from their current position
        cols.remove('evo_id_enc')
        cols.remove('evo_index_enc')
        # Insert them after enc_id (index 1)
        cols.insert(1, 'evo_id_enc')
        cols.insert(2, 'evo_index_enc')
        df = df[cols]
        
        print(f"  - Added evo_id_enc and evo_index_enc columns to {len(df)} encounters")
        return df
        
    except FileNotFoundError:
        print(f"  - Warning: Could not find stats file {stats_path}")
        print("  - Adding empty evo_id_enc and evo_index_enc columns")
        df = df.copy()
        df.insert(1, 'evo_id_enc', pd.NA)
        df.insert(2, 'evo_index_enc', pd.NA)
        return df


def add_exp_enc(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    def _norm(s):
        s = str(s).lower().replace(" ", "_")
        return "".join(ch for ch in s if ch in set("abcdefghijklmnopqrstuvwxyz0123456789_/")).replace("__", "_").rstrip("_")
    exp_types = pd.read_csv(Path(__file__).parent.parent / 'data/gen_all/exp_types.csv')
    exp_types["pokemon_norm"] = exp_types["pokemon"].map(_norm)
    exp_table = pd.read_csv(Path(__file__).parent.parent / 'data/gen_all/exp_table.csv')
    df_exp = exp_table.melt(id_vars=["Level"], var_name="exp_type", value_name="exp_enc").rename(columns={"Level": "level_enc"})
    df = df.copy()
    df["level_enc"] = pd.to_numeric(df["level_enc"], errors="coerce").astype("Int64")
    df = df.merge(exp_types[["pokemon_norm", "exp_type"]], left_on="pokemon_enc", right_on="pokemon_norm", how="left")
    df = df.merge(df_exp, left_on=["exp_type", "level_enc"], right_on=["exp_type", "level_enc"], how="left")
    df = df.drop(columns=["pokemon_norm","exp_type"])
    return df


def _apply_power_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    POWER_ADJ_CSV = Path(__file__).parent.parent / 'data/gen_all/move_power_adjustments.csv'
    try:
        _adj_raw = pd.read_csv(POWER_ADJ_CSV)
        _adj_raw = _adj_raw.rename(columns={"Name": "move_name", "new_power": "move_power"})
        # Use the same normalization as normalize_text_columns to match encounter move names
        _adj_raw = normalize_text_columns(_adj_raw[["move_name", "move_power"]])
        power_adj_map = dict(zip(_adj_raw.move_name, _adj_raw.move_power))
    except FileNotFoundError:
        power_adj_map = {}

    if not power_adj_map:
        return df

    df = df.copy()
    for i in range(1, 5):
        ncol = f"move_name_{i}_enc"
        pcol = f"move_power_{i}_enc"
        mask = df[ncol].isin(power_adj_map)
        df.loc[mask, pcol] = df.loc[mask, ncol].map(power_adj_map)
    return df


def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default config if file doesn't exist
        return {"level_calc_method (sequential_max/independent)": "independent"}

def apply_sequential_max_to_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sequential_max logic to encounters dataframe by modifying exp_enc values.
    For each row, exp_enc becomes the maximum of all exp_enc values at or before that stage.
    """
    print("  - Applying sequential_max logic to encounters...")
    
    # Sort by stage_enc and exp_enc to ensure proper order
    df = df.sort_values(['stage_enc', 'exp_enc'], ascending=[True, True]).reset_index(drop=True)
    
    # Apply sequential max logic: each exp_enc becomes max of all previous exp_enc at or before that stage
    max_exp_enc = 0
    for idx in df.index:
        current_exp_enc = df.loc[idx, 'exp_enc']
        if pd.notna(current_exp_enc):
            # Take max of previous max_exp_enc vs current exp_enc
            max_exp_enc = max(max_exp_enc, current_exp_enc)
            # Update exp_enc to reflect the sequential max
            df.loc[idx, 'exp_enc'] = max_exp_enc
    
    print(f"  - Sequential max applied to {len(df)} encounters")
    return df

def add_enc_id(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.sort_values(["stage_enc", "exp_enc"], ascending=[True, True])
        .reset_index(drop=True)
    )
    df.insert(0, "enc_id", range(1, len(df) + 1))
    return df


def generate_encounters(gen: int) -> pd.DataFrame:
    """
    Generate encounter data for a specific generation.
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        
    Returns:
        pd.DataFrame: Complete encounter data with stats, moves, and metadata
    """
    print(f'Generating encounters for gen {gen}')
    
    # Load configuration to check level calculation method
    config = load_config()
    level_calc_method = config.get("level_calc_method (sequential_max/independent)", "independent")
    
    df = load_trainer_data(gen)
    df = apply_location_adjustments(df)  # Apply location overrides before stage calculation
    df = add_stages(gen, df)
    df = calc_enc_stats(gen, df)
    enc_moves = calc_enc_moves(gen, df)
    df = add_encounter_moves(gen, df, enc_moves)
    df = _apply_power_adjustments(df)
    df = add_exp_enc(df)
    
    # Apply sequential_max logic if configured
    if level_calc_method == "sequential_max":
        print("  - Sequential_max level calculation method detected")
        df = apply_sequential_max_to_encounters(df)
    else:
        print("  - Using independent level calculation method")
    
    df = add_enc_id(df)
    
    # Add evolution information for encounters (evo_id_enc and evo_index_enc)
    df = add_evo_info_enc(gen, df)
    
    # Filter out duplicate rows based on pokemon_enc, stage_enc, level_enc, and move names
    initial_count = len(df)
    duplicate_columns = [
        'pokemon_enc', 'stage_enc', 'level_enc', 
        'move_name_1_enc', 'move_name_2_enc', 'move_name_3_enc', 'move_name_4_enc'
    ]
    df = df.drop_duplicates(subset=duplicate_columns, keep='first')
    final_count = len(df)
    duplicates_removed = initial_count - final_count
    
    if duplicates_removed > 0:
        print(f'  - Removed {duplicates_removed} duplicate encounters')
    
    print(f'Gen {gen} encounters complete')
    return df


# Optional: Test the function
if __name__ == "__main__":
    from pathlib import Path
    
    # Define intermediate_files directory
    intermediate_files_dir = Path(__file__).parent.parent / "intermediate_files"
    
    for gen in [1, 2, 3]:
        df = generate_encounters(gen)
        
        # Save to intermediate_files directory when run manually for debugging
        output_path = intermediate_files_dir / f'encounters_gen{gen}.csv'
        df.to_csv(output_path, index=False)
        print(f"\n*** DEBUG MODE: Gen {gen} encounters saved to {output_path} ***")
