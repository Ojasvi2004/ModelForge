from pymongo import MongoClient
import certifi

uri = "mongodb+srv://ojasvisaini2111_db_user:tIqzVBGaqkEUA8Rd@cluster0.jpqpoad.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, tlsCAFile=certifi.where())

print(client.admin.command("ping"))