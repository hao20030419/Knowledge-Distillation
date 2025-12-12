from GeminiAgent.pipeline.generate_dataset import generate_dataset
from GeminiAgent.pipeline.clean_dataset import clean_dataset

if __name__ == "__main__":
    
    print("🚀 Gemini Step 1：產生題目資料集...")
    generate_dataset(total=1, workers=1)
    '''
    print("\n🧹 Gemini Step 2：清洗資料集...")
    clean_dataset()
    '''