import os
import shutil

def main():
    # Files we are looking for
    required_files = ["xgb_model.pkl", "selected_features.pkl", "best_threshold.pkl"]
    
    # Destination directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory at: {models_dir}")
        
    print(f"Searching for model files in: {project_root} ...")
    
    found_count = 0
    for root, dirs, files in os.walk(project_root):
        # Don't search inside the models dir itself (avoids self-copying)
        if os.path.abspath(root) == os.path.abspath(models_dir):
            continue
            
        for file in files:
            if file in required_files:
                src = os.path.join(root, file)
                dst = os.path.join(models_dir, file)
                
                try:
                    shutil.copy2(src, dst)
                    print(f"✅ Found and moved: {file}")
                    print(f"   Source: {src}")
                    found_count += 1
                except Exception as e:
                    print(f"❌ Error moving {file}: {e}")
                    
    if found_count == 0:
        print("\n⚠️  No model files found!")
        print("You need to run your training notebook (training/Tiyani/exports/notebook.py) to generate 'xgb_model.pkl'.")
    else:
        print(f"\n🎉 Successfully moved {found_count} files to 'models/'.")
        print("👉 PLEASE RESTART YOUR BACKEND SERVER NOW.")

if __name__ == "__main__":
    main()