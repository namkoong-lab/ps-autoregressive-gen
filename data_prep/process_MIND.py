import os
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description='Processing MIND data into news article and click files')
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    args = parser.parse_args()
    print(args)
    path = args.data_dir

    # News Article Data ######################################
    news_dict = {}
    for dataset in ["train"]:
        print(dataset)
        with open( os.path.join(path, dataset, "news.tsv"), 'r') as f:
            for cnt, line in enumerate(f):
                if cnt % 100000 == 0:
                    print(cnt)
                line_items = line.strip().split('\t')

                news_id = line_items[0]
                category = line_items[1]
                subcategory = line_items[2]
                title = line_items[3]
                abstract = line_items[4]
                url = line_items[5]
                title_entities = line_items[6]
                abstract_entities = line_items[7]

                news_dict[news_id] = {
                        "category": category,
                        "subcategory": subcategory,
                        "title": title,
                        "abstract": abstract,
                        "url": url,
                        "title_entities": title_entities,
                        "abstract_entities": abstract_entities
                        }
        


    with open(os.path.join(path, "news_data_all.pt"), 'wb') as f:
        torch.save(news_dict, f)
    print("all", len(news_dict))


    # Process Click / No Click Data ######################################

    data_dict = {}
    for dataset in ["train"]:
        print(dataset)
        with open( os.path.join(path, dataset, "behaviors.tsv"), 'r') as f:
            for cnt, line in enumerate(f):
                if cnt % 100000 == 0:
                    print(cnt)
                line_items = line.strip().split("\t")
                
                impression_id = line_items[0]
                user_id = line_items[1]
                time = line_items[2]
                user_history = line_items[3]
                impressions = line_items[4]
              
                impressions_list = impressions.split(" ")
                news_id_click = [x.split("-") for x in impressions_list]
                
                for (news_id, click) in news_id_click:
                    if news_id not in data_dict.keys():
                        data_dict[news_id] = [(user_id, int(click))]
                    else:
                        data_dict[news_id].append( (user_id, int(click)) )
       

    with open(os.path.join(path, "click_data_all.pt"), 'wb') as f:
        torch.save(data_dict, f)

    print("all", len(data_dict))

if __name__ == "__main__":
    main()

