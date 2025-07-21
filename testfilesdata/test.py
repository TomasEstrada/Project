# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 13:29:11 2025

@author: testr
"""

import requests
url = "http://localhost:5477/blackrock/challenge/v1/returns:ppr"
#%%%
# Sample payload with your features
payload = {'age':30,
           'inflation':4.83,
    "wage": 75000,
    "transactions": [
    {
        "date": "2025-01-01 00:00",
        "amount": 59307
    },
    {
        "date": "2025-01-13 01:26",
        "amount": 16544
    },
    {
        "date": "2025-09-23 21:50",
        "amount": 5978
    },
    {
        "date": "2025-03-21 08:09",
        "amount": 93246
    },
    {
        "date": "2025-02-13 10:33",
        "amount": 48120
    },
    {
        "date": "2025-11-09 05:02",
        "amount": 5408
    },

    ],
    "p": [
    {
        "extra": 10,
        "start": "2025-08-11 00:00",
        "end": "2025-09-19 14:52"
    },
    {
        "extra": 5,
        "start": "2025-12-10 00:00",
        "end": "2025-12-21 14:52"
    },
    {
        "extra": 7,
        "start": "2025-01-11 00:00",
        "end": "2025-01-15 14:52"
    },
    {
        "extra": 7,
        "start": "2025-12-11 00:00",
        "end": "2025-12-15 14:52"
    }
],
    "q": [

    {
        "fixed": 10,
        "start": "2025-04-11 00:00",
        "end": "2025-05-19 14:52"
    },

],
    "k": [

    {
        "start": "2025-01-01 00:00",
        "end": "2025-12-21 14:52"
    }
]
}

try:
    # Send POST request
    response = requests.post(url, json=payload)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse and print response JSON
        result = response.json()
        print("API response:", result)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
except Exception as e:
    print("Error during request:", e)
    
#%%

url = "http://localhost:5477/blackrock/challenge/v1/performance"
try:
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse and print response JSON
        result = response.json()
        print("API response:", result)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
except Exception as e:
    print("Error during request:", e)
    