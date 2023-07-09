from sqlalchemy import Column, String, Boolean, Integer
from data_base.dbcore import Base


class User(Base):
    __tablename__ = 'users'

    chat_id = Column(Integer, primary_key=True)
    is_transfer_mode = Column(Boolean)
    style_image_path = Column(String)

    def __str__(self):
        return self.name
