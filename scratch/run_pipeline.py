import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    print(f"=== RUNNING NOTEBOOK: {notebook_path} ===")
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} does not exist!")
        return False
        
    try:
        # Load the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            
        # Configure execution preprocessor (timeout is 1200 seconds per cell)
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
        
        # Run the notebook setting the working directory to notebooks/
        ep.preprocess(nb, {'metadata': {'path': 'notebooks'}})
        
        # Save the executed notebook back
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        print(f"✅ Finished {notebook_path} successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error executing {notebook_path}:")
        print(e)
        return False

def main():
    # Run the fishes transformer model notebook first
    success = run_notebook("notebooks/06_transformer_hybrid_model.ipynb")
    if not success:
        print("Pipeline execution failed at Notebook 06.")
        sys.exit(1)
        
    # Run the hybrid fusion and ensembling notebook next
    success = run_notebook("notebooks/05_hybrid_fusion_and_ensembling.ipynb")
    if not success:
        print("Pipeline execution failed at Notebook 05.")
        sys.exit(1)
        
    print("🎉 Pipeline executed successfully! Submission file is ready in submissions/submission.csv.")

if __name__ == "__main__":
    main()
