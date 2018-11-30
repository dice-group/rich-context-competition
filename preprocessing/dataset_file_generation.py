import json, os

processed_files_path = "project/additional_files/processed_articles"
main_content_path = "project/additional_files/main_content" #abstract and data/methods
other_content_path = "project/additional_files/other_content"

if not os.path.exists(main_content_path):
    os.makedirs(main_content_path)

if not os.path.exists(other_content_path):
    os.makedirs(other_content_path)

for filename in os.listdir(processed_files_path):
    with open(os.path.join(processed_files_path, filename)) as f:
        print(filename)
        data = json.load(f)
        main_text = []
        other_text = []
        for key in data:
            if "abstract" == key:
                main_text.append(data["abstract"])
            elif "methodology" == key:
                main_text.append(data["methodology"])
            elif "subject" == key:
                continue
            elif "keywords" == key:
                other_text.append(" ".join(data["keywords"]))
            else:
                other_text.append(data[key])

        with open(os.path.join(main_content_path, filename), "w") as f:
            f.write("\n".join(main_text))

        with open(os.path.join(other_content_path, filename), "w") as f:
            f.write("\n".join(other_text))
        # text = data["abstract"]
print("done.")


