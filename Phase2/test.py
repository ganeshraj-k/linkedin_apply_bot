import json





with open('userdata.json' , 'r') as f:
    data=json.load(f)

print(data['linkedin_username'])
