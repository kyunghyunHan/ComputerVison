import yaml

data = {
    "train" : '/tld_sample/train/',
        # "val" : '/tld_sample/valid/',
        # "test" : '/tld_sample/test/', optional 
        "names" : {0 : 'red', 1 : 'green'}}

with open('./tld.yaml', 'w') as f :
    yaml.dump(data, f)

# check written file
with open('./tld.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)
