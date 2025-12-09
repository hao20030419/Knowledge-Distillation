from GPTagent.pipeline.generate_dataset import generate_dataset
from GPTagent.pipeline.clean_dataset import clean_dataset

if __name__ == "__main__":
    '''
    print("🚀 Step 1：產生題目資料集...")
    generate_dataset(total=20, workers=4)
    '''
    print("\n🧹 Step 2：清洗資料集...")
    clean_dataset("dataset.jsonl")

    print("\n🎉 All Done!")