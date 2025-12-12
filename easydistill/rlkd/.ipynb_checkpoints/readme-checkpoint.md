### 需要重写

def get_index(temp) -> int:
    pass

def get_answer(temp) -> str:
    pass

def get_input(temp) -> str:
    pass
    
def get_reward(predict_str: str, ground_truth: str, ...) -> float:
    pass
    
### 需要配置模型或key

此处img是直接输入的PIL对象图像

def student(img,query):
    pass
    
def teacher(img,query):
    pass