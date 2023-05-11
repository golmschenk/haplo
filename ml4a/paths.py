from pathlib import Path

from appdirs import user_data_dir

user_data_directory = Path(user_data_dir(appname='haplo', appauthor='golmschenk'))
