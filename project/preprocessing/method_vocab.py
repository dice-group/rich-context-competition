from project.preprocessing.preprocess_publications import PublicationPreprocessing
import json

class Helper_MethodExtractor:
    """
    generates noun phrases for the methods vocabulary
    """
    def __init__(self):
        self.pp = PublicationPreprocessing()

    def extract_np_for_dbpedia_vocab(self, path = "project/train_test/dbpedia_method_vocab.json"):
        result = {}

        with open(path) as file:
            data = json.load(file)
            for key in data:
                obj_list = []
                for obj in data[key]:
                    sub_dict = {}
                    for k in obj:
                        if k == "abs":
                            sub_dict["abs_np"] = self.pp.gather_nounPhrases(obj[k])
                        sub_dict[k] = obj[k]
                    obj_list.append(sub_dict)
                result[key] = obj_list

        with open("project/train_test/dbpedia_methods_np_vocab.json", "w") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

        print('file written.')



    def extract_np_for_vocab(self, path = "project/train_test/sage_research_methods.json"):
        result = {}

        with open(path) as file:
            data = json.load(file)
            result["@context"] = data['@context']
            result["@graph"] = []

            for val in data["@graph"]:
                sub_dict = {}
                for key in val:
                    if key == "skos:definition":
                        sub_dict["@nounchunks"] = self.pp.gather_nounPhrases(val[key]["@value"])

                    sub_dict[key] = val[key]
                result["@graph"].append(sub_dict)

        with open("project/train_test/methods_vocab.json", "w") as file:
            json.dump(result, file, indent=4)

        print('file written.')


def main():
    obj = Helper_MethodExtractor()
    # obj.extract_np_for_vocab()
    obj.extract_np_for_dbpedia_vocab()

if __name__ == '__main__':
    main()
