from pymongo import MongoClient

# MongoDB client and database initialization
client = MongoClient('mongodb+srv://ResumeParser:12345@cluster0.db4ll.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['db1']
collection = db['resume_details']
