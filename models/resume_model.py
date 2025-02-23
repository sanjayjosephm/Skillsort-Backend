from models.database import collection


def add_resume(resume_data):
    """
    Adds a resume to the collection.
    :param resume_data: Dictionary containing resume details.
    """
    try:
        collection.insert_one(resume_data)
        return {"status": "success", "message": "Resume added successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_all_resumes():
    """
    Retrieves all resumes from the collection.
    """
    try:
        resumes = list(collection.find({}, {"_id": 0}))  # Exclude ObjectID for simplicity
        return {"status": "success", "resumes": resumes}
    except Exception as e:
        return {"status": "error", "message": str(e)}