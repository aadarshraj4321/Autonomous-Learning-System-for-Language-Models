import pandas as pd
from db_utils import get_all_feedback_from_db
from datasets import Dataset, DatasetDict



HUGGING_FACE_REPO_NAME = "boyinfuture/autonomous-learning-dataset"




def create_preference_dataset():
    print("Getting feedback fromd database")
    feedback_list = get_all_feedback_from_db()
    
    if not feedback_list:
        print("No feedback inside the database")
        return
    
    df = pd.DataFrame([vars(f) for f in feedback_list])
    print(f"Total {len(df)} feedback records found")

    df_approved = df.copy()

    preference_data = []

    for index, row in df_approved[df_approved['feedback_type'] == 'down'].iterrows():
        if row['correction_text'] and row['correction_text'].strip():
            preference_data.append({
                'prompt': row['prompt'],
                'chosen': row['correction_text'],
                'rejected': row['response']
            })
    
    if not preference_data:
        print("There is no valid preference pairs for training")
        return
    
    print(f"{len(preference_data)} preference pair created")

    hf_dataset = Dataset.from_pandas(pd.DataFrame(preference_data))

    train_test_split = hf_dataset.train_test_split(test_size=0.1)


    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })


    print("Huggine Face dataset dicionary created")
    print(dataset_dict)



    print(f"Push Dataset to '{HUGGING_FACE_REPO_NAME}' going on")
    dataset_dict.push_to_hub(HUGGING_FACE_REPO_NAME, private=True)
    print("Dataset successfully pushed to HuggingFace")



if __name__ == "__main__":
    create_preference_dataset()