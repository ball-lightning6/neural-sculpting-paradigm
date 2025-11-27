import os
import shutil

base_dir = r"e:\code\neural-sculpting-paradigm\to_be_organized"

categories = {
    "symbolic_math_logic": [
        "add", "multiply", "modulo", "evaluate", "deduction", "sat_solver", 
        "rsa_encryption", "simple_block_cipher", "check_boolean_equivalence",
        "count_set_bits", "min_flips", "min_swaps"
    ],
    "cellular_automata": [
        "cellular_automata", "game_of_life", "reaction_diffusion"
    ],
    "algorithms": [
        "sort", "edit_distance", "hanoi", "maze", "trapping_rain_water", "skyline",
        "shortest_path", "blocks_world", "sudoku", "valid_parentheses", "river_crossing",
        "freedom_trail", "matrix_flip", "longest_subsequence", "find_articulation_points",
        "sum_of_subset", "min_k_bit_flips"
    ],
    "games": [
        "checkers", "chess", "nim_game", "treasure_hunt"
    ],
    "physics_simulation": [
        "projectile", "catenary", "orbital", "snell", "cube_rotation"
    ],
    "visual_reasoning_arc": [
        "arc_", "triangle", "polygon", "count_shapes", "line_angle", "coords_to",
        "checkerboard", "symbolic_to_image", "convolution_2d"
    ],
    "function_approx": [
        "sin_function", "polynomial_shift"
    ],
    "scripts_utils": [
        "create_videos", "preprocess_legal_moves"
    ]
}

# Fallback for files not matching specific patterns but clearly belonging to a group based on keywords
# The order matters: specific prefixes first

def organize():
    # Create directories
    for category in categories:
        dir_path = os.path.join(base_dir, category)
        os.makedirs(dir_path, exist_ok=True)

    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    
    for filename in files:
        moved = False
        lower_name = filename.lower()
        
        # Custom logic for some ambiguous ones
        if "arc_" in lower_name: # Priority for ARC
            shutil.move(os.path.join(base_dir, filename), os.path.join(base_dir, "visual_reasoning_arc", filename))
            continue

        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in lower_name:
                    shutil.move(os.path.join(base_dir, filename), os.path.join(base_dir, category, filename))
                    moved = True
                    break
            if moved:
                break
        
        if not moved:
            print(f"Skipped: {filename}")

if __name__ == "__main__":
    organize()
    print("Organization complete.")