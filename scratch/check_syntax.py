import json
import traceback

notebook_path = "notebooks/05_hybrid_fusion_and_ensembling.ipynb"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print(f"Loaded notebook: {notebook_path}")
    print(f"Number of cells: {len(nb['cells'])}")
    
    for i, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type", "")
        if cell_type == "code":
            source = "".join(cell.get("source", []))
            print(f"\n--- Checking Code Cell {i} ---")
            try:
                compile(source, f"Cell_{i}", "exec")
                print(f"✅ Cell {i} syntax is OK.")
            except SyntaxError as e:
                print(f"❌ Cell {i} SyntaxError:")
                traceback.print_exc()
                # Print the source code with line numbers to inspect
                print("Source code:")
                for line_no, line in enumerate(cell.get("source", []), 1):
                    print(f"{line_no:4d}: {line}", end="")
                print()
except Exception as e:
    print(f"Error loading notebook: {e}")
    traceback.print_exc()
