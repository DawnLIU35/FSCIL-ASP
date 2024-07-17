def get_model(model_name, args):
    name = model_name.lower()
    if name == "asp":
        from models.asp import Learner 
    else:
        assert 0
    
    return Learner(args)
