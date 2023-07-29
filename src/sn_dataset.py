from torch.utils.data import Dataset

from code_embedding import CodeEmbedding


class SNDataset(Dataset):
    def __init__(
        self, name_to_reference_map: dict, positive_negative_function_map: dict
    ):
        self.name_to_reference_map = name_to_reference_map
        self.positive_negative_function_map = positive_negative_function_map
        self.CodeEmbedding = CodeEmbedding()
        self.samples = []
        self._find_unique_combinations(
            name_to_reference_map, positive_negative_function_map
        )
        self.current_item = -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Get next item from the combination.
        self.current_item += 1
        anchor_fun, positive_fun, negative_fun = self.samples[self.current_item]
        anchor = self.CodeEmbedding.getPerfectFunctionEmbedding(anchor_fun)
        positive = self.CodeEmbedding.getPerfectFunctionEmbedding(positive_fun)
        negative = self.CodeEmbedding.getPerfectFunctionEmbedding(negative_fun)
        return anchor, positive, negative

    def _find_unique_combinations(
        self, name_to_reference_map: dict, positive_negative_function_map: dict
    ):
        combinations = []
        for key in name_to_reference_map:
            if key in positive_negative_function_map:
                value1 = name_to_reference_map[key]
                value2, value3 = positive_negative_function_map[key]

                for v2 in value2:
                    for v3 in value3:
                        combinations.append((value1, v2, v3))
        self.samples = combinations
