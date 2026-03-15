import sqlalchemy as sql
import sqlalchemy.orm as orm
from sqlalchemy.ext.hybrid import hybrid_property

import hashlib

db_url = "sqlite:///database.db"
engine = sql.create_engine(db_url)
Base = orm.declarative_base()
Meta = Base.metadata
Session = orm.sessionmaker(engine)

class User(Base):
    __tablename__ = "users"

    id: orm.Mapped[int] = orm.mapped_column(sql.Integer, nullable=False, primary_key=True)
    
    name: orm.Mapped[str] = orm.mapped_column(sql.String, nullable=False, default="Name")
    surname: orm.Mapped[str] = orm.mapped_column(sql.String, nullable=False, default="Surname")

    username: orm.Mapped[str] = orm.mapped_column(sql.String, nullable=False, unique=True)
    password_hash: orm.Mapped[str] = orm.mapped_column(sql.String, nullable=False)

    is_admin: orm.Mapped[bool] = orm.mapped_column(sql.Boolean, default=False)

    def _get_hex_of_value(value: str) -> str:
        value_hash = hashlib.sha256(value.encode("utf-8"))
        value_hex = value_hash.hexdigest()
        return value_hex
    
    def set_password(self, password: str) -> str:
        self.password_hash = self._get_hex_of_value(password)
        return self.password_hash
    
    def check_password(self, for_check: str) -> bool:
        fchhex = self._get_hex_of_value(for_check)
        return fchhex == self.password_hash

Meta.create_all(engine)