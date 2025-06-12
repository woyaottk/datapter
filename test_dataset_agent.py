import os
from dotenv import load_dotenv
load_dotenv('.env')

from src.domain.dataset_agent import main

if __name__ == '__main__':
    main()
