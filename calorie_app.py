import os
import time
import json
import yaml
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# temp_json = """
# {
#   "food_name": "risotto",
#   "food_components": [
#     { "ingredient": "rice", "quantity": "100", "unit": "grams", 'query_param': ['rice', 'jamine rice', 'basmati rice', 'brown rice'] },
#     { "ingredient": "shrimp", "quantity": "50", "unit": "grams" },
#     { "ingredient": "squid", "quantity": "50", "unit": "grams" },
#     { "ingredient": "clams", "quantity": "50", "unit": "grams" },
#     { "ingredient": "onion", "quantity": "30", "unit": "grams" },
#     { "ingredient": "garlic", "quantity": "5", "unit": "grams" },
#     { "ingredient": "broth", "quantity": "300", "unit": "milliliters" },
#     { "ingredient": "parsley", "quantity": "5", "unit": "grams" },
#     { "ingredient": "olive oil", "quantity": "10", "unit": "grams" },
#     { "ingredient": "cheese", "quantity": "20", "unit": "grams" }
#   ]
# }
# """

class LLMModelInferencing:
    
    @staticmethod
    def get_ingredients(image_path):
        url = os.path.join(os.environ.get("BASE_URL"), 'analyze_food')
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("✅ Response:")
                return str(response.json())
            else:
                return None
    

class RAG:
    
    def __init__(self):
        self.pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = PineconeVectorStore(index=self.__get_index__(), embedding=self.embeddings)
        
    def __get_index__(self):
        index_name = "calorie-index"
        existing_indexes = [index_info["name"] for index_info in self.pinecone.list_indexes()]
        if index_name not in existing_indexes:
            self.pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                deletion_protection="enabled",
            )
            while not self.pinecone.describe_index(index_name).status["ready"]:
                time.sleep(1)
        return self.pinecone.Index(index_name)
    
    def __document_loader__(self, file_paths: list):
        documents = []
        for path in file_paths:
            if os.path.exists(path):
                try:
                    if path.endswith(".csv"):
                        loader = CSVLoader(path)
                    pages = loader.load_and_split()
                    documents.extend(pages)  # Use extend instead of append
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"File not found: {path}")
        return documents
    
    def store_in_vectorstore(self, file_paths: list):
        documents = self.__document_loader__(file_paths)
        split_documents = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(split_documents)
        
    def query_vectorstore(self, ingredients: list):
        neutrients = []
        scores = []
        for item in ingredients:
            result = self.vectorstore.similarity_search_with_score(item.lower(), k=1)[0]
            if result:
                doc, score = result
                neutrients.append(doc.page_content.split("description: ")[-1])
                scores.append(score)
        if neutrients:
            return neutrients[scores.index(max(scores))]
        else:
            return None    
    
class PromptEngineering:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=os.environ.get("OPENAI_MODEL_NAME"))
        self.prompt = yaml.safe_load(open("prompts.yaml", "r"))
    
    def __use_llm__(self, prompt):
        return self.llm.bind(response_format={"type": "json_object"}).invoke(prompt).content
    
    def __unit_correction__(self, food):
        unit_list = ['milliliters', 'ml']
        if food['unit'] in unit_list:
            food['unit'] = 'gm' #taking water as a sample -> Improve your app. You can introduce viscosity/density
            food['quantity'] = (int(food['quantity']))/1000
        return food
    
    def __get_similar_food__(self, similar_food: dict, food_name: str) -> list[str]:
        return similar_food[food_name]
    
    def query_prep(self, model_response: str):
        chef_response = json.loads(model_response)
        food_name = chef_response["food_name"]
        food_components = [self.__unit_correction__(component) for component in chef_response["food_components"]]
        ingredients_list = [item['ingredient'] for item in food_components]
        try:
            similar_food = json.loads(self.__use_llm__(self.prompt["search_ingredients"].format(ingredients_list=ingredients_list)))
            food_components = [
                {**component, 'query_param':  [component['ingredient']] + self.__get_similar_food__(similar_food, component['ingredient'])}
                for component in food_components
            ]
            return food_name, food_components
        except Exception as e:
            return None, "JSON decoding error"
        
    def __nutrients_formatting__(self, food_components):
        nutrient_list = None
        merged = []
        ingredient_nutrients = [
            {'ingredient': item['ingredient'], 'nutrient_description': item['nutrient_description']}
            for item in food_components
        ]
        try:
            nutrients = json.loads(self.__use_llm__(self.prompt["JSON_formatter"].format(nutrient_description=ingredient_nutrients)))
            for value in nutrients.values():
                nutrient_list = value
                break
            nutrient_lookup = {item['ingredient'].lower(): item['nutrients'] for item in nutrient_list}
            for item in food_components:
                ing_name = item['ingredient'].lower()
                nutrients = nutrient_lookup.get(ing_name)
                merged.append({
                    "ingredient": item["ingredient"],
                    "quantity": item["quantity"],
                    "unit": item["unit"],
                    "nutrients": nutrients
                })
            return merged
        except Exception as e:
            return None
        # print(ingredient_nutrients)
    
    def get_food_nutrients(self, food_components, food_name):
        total_nutrients = {}
        food_formatted_list = self.__nutrients_formatting__(food_components)
        for item in food_formatted_list:
            nutrients = item.get("nutrients")
            quantity = float(item.get("quantity"))

            scaling_factor = quantity / 100.0
            for nutrient, value in nutrients.items():
                scaled_value = value * scaling_factor
                total_nutrients[nutrient] = total_nutrients.get(nutrient, 0) + scaled_value
            
        flat_total = {f"total_{k}": round(v, 4) for k, v in total_nutrients.items()}
        flat_total["food_name"] = food_name
        return flat_total
    

def run_rag():
    file_paths = ["./data/nutrients.csv"]
    RAG().store_in_vectorstore(file_paths)

    
def print_progress(percent, message):
    print(f"[{percent}%] {message}")

def run_calorie_app(image_path, update_progress_callback):
    # Step 1: Image processing
    update_progress_callback(0, "Starting image processing...")
    raw_response = LLMModelInferencing.get_ingredients(image_path)
    response = raw_response.replace("'", '"')
    print(f"Food and its ingredients have been detected by LLM model \n{response}",)
    update_progress_callback(50, "Image processing complete.")

    # Step 2: PromptEngineering init
    pe = PromptEngineering()
    update_progress_callback(52, "Initialized Prompt Engineering.")

    # Step 3: Prompt prep
    if response:
        food_name, food_components = pe.query_prep(response)
        update_progress_callback(62, "Prompt processed and food components extracted.")

        # Step 4: RAG enrichment
        enriched_components = []
        for i, ingredient in enumerate(food_components):
            desc = RAG().query_vectorstore(ingredient['query_param'])
            enriched_components.append({**ingredient, 'nutrient_description': desc})
            progress = 62 + int((i + 1) / len(food_components) * 30)
            update_progress_callback(progress, f"Enriching ingredient {i + 1}/{len(food_components)}")

        food_components = enriched_components
        update_progress_callback(95, "RAG enrichment complete.")

        # Step 5: Final nutrient composition
        result = pe.get_food_nutrients(food_components, food_name)
        update_progress_callback(100, "Nutrient breakdown complete.")
        return result
    else:
        update_progress_callback(100, "Error in image processing.")
        return None
    

if __name__ == "__main__":
    # run_rag()
    print(run_calorie_app('food.jpg', print_progress)) 
    
    
    
        
    
    
    
    