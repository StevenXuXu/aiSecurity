import requests

# 请求的URL
url = "http://127.0.0.1:5000/detect"

# 请求的参数
params = {
    'user' : "我是谁？",
    'assistant' : "你是谁？"
}

# 发送POST请求
response = requests.post(url, params=params)

# 输出返回结果
print(response.json())