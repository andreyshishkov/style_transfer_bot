import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get('TOKEN')

NAME_DB = 'users.db'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join('sqlite:///' + BASE_DIR, NAME_DB)
