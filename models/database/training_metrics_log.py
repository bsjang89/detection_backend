from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from config.database import Base


class TrainingMetricsLog(Base):
    __tablename__ = "training_metrics_log"

    id = Column(Integer, primary_key=True, index=True)
    training_session_id = Column(Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False)

    epoch = Column(Integer, nullable=False)

    # Loss metrics
    train_box_loss = Column(Float)
    train_cls_loss = Column(Float)
    train_dfl_loss = Column(Float)
    train_total_loss = Column(Float)

    val_box_loss = Column(Float)
    val_cls_loss = Column(Float)
    val_dfl_loss = Column(Float)
    val_total_loss = Column(Float)

    # Performance metrics
    map50 = Column(Float)
    map50_95 = Column(Float)
    precision = Column(Float)
    recall = Column(Float)

    # Learning rate
    learning_rate = Column(Float)

    # Timestamp
    logged_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    training_session = relationship("TrainingSession", back_populates="metrics_logs")
