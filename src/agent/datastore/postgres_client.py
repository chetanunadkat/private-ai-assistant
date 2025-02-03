# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Manager conversation history return relavant conversation history
based on session_id using postgres database
"""

from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Float, JSONB, func, insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

from src.common.utils import get_config

# TODO: Move this config to __init__ method
db_user = os.environ.get("POSTGRES_USER")
db_password = os.environ.get("POSTGRES_PASSWORD")
db_name = os.environ.get("POSTGRES_DB")

settings = get_config()
# Postgres connection URL
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{settings.database.url}/{db_name}?sslmode=disable"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class ConversationHistory(Base):
    __tablename__ = 'conversation_history'
    
    session_id = Column(String(255), primary_key=True)
    conversation_hist = Column(JSONB)
    create_time = Column(DateTime, default=func.now())
    last_accessed = Column(DateTime, default=func.now(), onupdate=func.now())

class PostgresClient:
    def __init__(self):
        self.engine = engine
        Base.metadata.create_all(self.engine)

    def store_conversation(self, session_id: str, conversation_history: list) -> bool:
        session = Session()
        try:
            stmt = insert(ConversationHistory).values(
                session_id=session_id,
                conversation_hist=conversation_history,
                last_accessed=func.now()
            ).on_conflict_do_update(
                index_elements=[ConversationHistory.session_id],
                set_={
                    ConversationHistory.conversation_hist: conversation_history,
                    ConversationHistory.last_accessed: func.now()
                }
            )
            session.execute(stmt)
            session.commit()
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def fetch_conversation(self, session_id: str):
        session = Session()
        try:
            conversation = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if conversation:
                return {
                    'session_id': conversation.session_id,
                    'user_id': conversation.user_id,
                    'last_conversation_time': conversation.last_conversation_time,
                    'conversation_history': json.loads(conversation.conversation_data)
                }
            return None
        except Exception as e:
            print(f"Error fetching conversation: {e}")
            return None
        finally:
            session.close()

    def delete_conversation(self, session_id: str):
        session = Session()
        try:
            conversation = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if conversation:
                session.delete(conversation)
                session.commit()
                print(f"Conversation with session_id {session_id} deleted successfully.")
            else:
                print(f"No conversation found with session_id {session_id}.")
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            session.rollback()
        finally:
            session.close()

    def is_session(self, session_id: str) -> bool:
        session = Session()
        try:
            exists = session.query(ConversationHistory).filter_by(session_id=session_id).first() is not None
            return exists
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return False
        finally:
            session.close()
