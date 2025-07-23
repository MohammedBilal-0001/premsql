from pathlib import Path
from typing import Optional, Union

from datasets import load_dataset
from tqdm.auto import tqdm

from premsql.datasets.base import (
    SupervisedDatasetForTraining,
    Text2SQLBaseDataset,
    Text2SQLBaseInstance,
)
from premsql.logger import setup_console_logger
from premsql.prompts import BASE_TEXT2SQL_PROMPT
from premsql.utils import filter_options, save_to_json

logger = setup_console_logger("[ARABIC-DATASET]")


class ArabicInstance(Text2SQLBaseInstance):
    def __init__(self, dataset: list[dict]) -> None:
        # First rename the fields before calling parent constructor
        for item in dataset:
            item["question"] = item.get("instruction", "")
            item["SQL"] = item.get("sql", "")
            item["db_path"] = "/content/drive/MyDrive/Text_2_SQL/MB/LlamaFactory/NorthWind/DB/northwind_simplified.db"
            item["db_id"]="northwind_simplified.db"
        super().__init__(dataset)

    def apply_prompt(
        self,
        num_fewshot: Optional[int] = None,
        prompt_template: Optional[str] = BASE_TEXT2SQL_PROMPT,
    ):
        prompt_template = (
            BASE_TEXT2SQL_PROMPT if prompt_template is None else prompt_template
        )
        for blob in tqdm(self.dataset, total=len(self.dataset), desc="Applying prompt"):
            few_shot_prompt = (
                ""
                if num_fewshot is None
                else self.add_few_shot_examples(db_id=blob["db_id"], k=num_fewshot)
            )
            final_prompt = prompt_template.format(
                schemas=blob["input"],
                additional_knowledge="",
                few_shot_examples=few_shot_prompt,
                question=blob["question"],  # Now using the renamed field
            )
            blob["prompt"] = final_prompt
        return self.dataset


class ArabicDataset(Text2SQLBaseDataset):
    def __init__(
        self,
        split: Optional[str] = "train",
        dataset_folder: Optional[Union[str, Path]] = "./data",
        hf_token: Optional[str] = None,
        force_download: Optional[bool] = False,
        json_file_name: Optional[str] = "data.json"
    ):
        dataset_folder = Path(dataset_folder)
        dataset_path = dataset_folder 
        if not dataset_path.exists() or force_download:
             raise FileNotFoundError(f"Dataset or file `{json_file_name}` not found in {dataset_path}")

        super().__init__(
            split="train",
            dataset_path=dataset_path,
            database_folder_name=None,
            json_file_name=json_file_name,
        )

    def setup_dataset(
        self,
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        model_name_or_path: Optional[str] = None,
        prompt_template: Optional[str] = BASE_TEXT2SQL_PROMPT,
        tokenize: Optional[bool]= False,
    ):
        if filter_by:
            self.dataset = filter_options(data=self.dataset, filter_by=filter_by)

        if num_rows:
            self.dataset = self.dataset[:num_rows]

        self.dataset = ArabicInstance(dataset=self.dataset).apply_prompt(
            num_fewshot=num_fewshot, prompt_template=prompt_template
        )

        return SupervisedDatasetForTraining(
            dataset=self.dataset,
            model_name_or_path=model_name_or_path,
            hf_token=self.hf_token,
            tokenize=tokenize,
        )
