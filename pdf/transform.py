import os

from pdf.storage import Storage
from pdf.tools import file_results, get_config, get_data_dir, ResultSerializer, FileConfig


def transfer_storage(src: Storage, dest: Storage):
    config = get_config()

    total = len(config)
    current = 0
    src_text_model = src.get_text_model()
    dest_text_model = dest.get_text_model()
    src_result_model = src.get_result_model()
    dest_result_model = dest.get_result_model()
    src_map_indices_model = src.get_map_indices_model()
    dest_map_indices_model = dest.get_map_indices_model()
    # src_similarity_model = src.get_similarity_model()
    # dest_similarity_model = dest.get_similarity_model()

    for file_config in config.values():
        index = file_config["index"]

        if src_text_model.exists(index):
            data = src_text_model.get(index)
            dest_text_model.save(index, data)

        if src_result_model.exists(index):
            data = src_result_model.get(index)
            dest_result_model.save(data)

        if src_map_indices_model.exists(index):
            data = src_map_indices_model.get(index)
            dest_map_indices_model.save(index, data)

        current += 1
        print(f"Finished {current}/{total}")


def transform_results(deserializer: ResultSerializer, serializer: ResultSerializer) -> None:
    config: FileConfig = get_config()
    data_dir = get_data_dir()
    total = len(config)
    current = 0

    for result in file_results(config, data_dir, serializer=deserializer):
        config_index = config[result["file"]]["index"]
        new_result = serializer.serialize(result)

        path = os.path.join(data_dir, f"{config_index}.json")

        with open(path, "w", encoding="utf8") as file:
            file.write(new_result)

        current += 1
        print(f"Finished transforming {current}/{total}")

    print("Finished transforming all")
