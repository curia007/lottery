from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Sample RESTful Web Service")


class Item(BaseModel):
    id: int
    name: str
    price: float


items: List[Item] = [
    Item(id=1, name="Laptop", price=999.99),
    Item(id=2, name="Mouse", price=24.99),
]


@app.get("/")
def home():
    return {"message": "Python RESTful Web Service is running"}


@app.get("/items")
def get_items():
    return items


@app.get("/items/{item_id}")
def get_item(item_id: int):
    for item in items:
        if item.id == item_id:
            return item

    raise HTTPException(status_code=404, detail="Item not found")


@app.post("/items")
def create_item(item: Item):
    for existing_item in items:
        if existing_item.id == item.id:
            raise HTTPException(status_code=400, detail="Item ID already exists")

    items.append(item)
    return {"message": "Item created", "item": item}


@app.put("/items/{item_id}")
def update_item(item_id: int, updated_item: Item):
    for index, item in enumerate(items):
        if item.id == item_id:
            items[index] = updated_item
            return {"message": "Item updated", "item": updated_item}

    raise HTTPException(status_code=404, detail="Item not found")


@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    for item in items:
        if item.id == item_id:
            items.remove(item)
            return {"message": "Item deleted"}

    raise HTTPException(status_code=404, detail="Item not found")
