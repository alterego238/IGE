# config your api key here

def get_api_key_for_index(index):
    api_keys = [
        
    ]
    return api_keys[index % len(api_keys)]

def get_base_url_and_api_key():
    base_url = ""
    api_key = ""
    return base_url, api_key