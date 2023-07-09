from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from os import path
from data_base.dbcore import Base
from settings.config import DATABASE


from data_base.models.user import User


class Singleton(type):

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs,)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class DBManager(metaclass=Singleton):

    def __init__(self):
        self.engine = create_engine(DATABASE)
        session = sessionmaker(bind=self.engine)
        self._session = session()
        if not path.isfile(DATABASE):
            Base.metadata.create_all(self.engine)

    def close(self):
        self._session.close()

    def get_all_chat_ids(self):
        result = self._session.query(User.chat_id).all()
        result = [x[0] for x in result]
        self.close()
        return result

    def get_curr_mode(self, chat_id: int):
        result = self._session.query(User.is_transfer_mode).filter_by(chat_id=chat_id).one()
        self.close()
        return result.is_transfer_mode

    def insert_chat_id(self, chat_id: int):
        user = User(
            chat_id=chat_id,
            is_transfer_mode=True,
            style_image_path='styles/default.jpg',
        )
        self._session.add(user)
        self._session.commit()
        self.close()

    def change_mode(self, chat_id: int, state: bool):
        self._session.query(User).filter_by(chat_id=chat_id).update(
            {'is_transfer_mode': state}
        )
        self._session.commit()
        self.close()

    def change_path(self, chat_id: int, style_path: str):
        self._session.query(User).filter_by(chat_id=chat_id).update(
            {'style_image_path': style_path}
        )

    def get_curr_style(self, chat_id):
        result = self._session.query(User.style_image_path).filter_by(chat_id=chat_id).one()
        self.close()
        return result.style_image_path
