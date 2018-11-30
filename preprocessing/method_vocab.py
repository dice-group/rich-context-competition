from project.preprocessing.preprocess_publications import PublicationPreprocessing
import json

vocab_path = "project/train_test/sage_research_methods.json"

class Helper_MethodExtractor:
    def __init__(self):
        self.pp = PublicationPreprocessing()

    def extract_np_for_vocab(self, path = vocab_path):
        result = {}
        with open(path) as file:
            data = json.load(file)
            result["@context"] = data['@context']
            result["@graph"] = []
            for val in data["@graph"]:
                #print(val)
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
    obj.extract_np_for_vocab()

if __name__ == '__main__':
    main()
