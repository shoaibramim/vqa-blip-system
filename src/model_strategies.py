"""
model_strategies.py

Strategy Pattern for VQA model backends.

Defines an abstract ModelStrategy interface and two concrete implementations:

    BLIPStrategy  -- Fine-tunes Salesforce/blip-vqa-base for multi-class
                     classification by replacing its generative head with a
                     linear classifier on top of the joint vision-language
                     encoder output.

    CLIPStrategy  -- Uses openai/clip-vit-base-patch32 with a frozen backbone
                     and a trainable MLP fusion head that concatenates the image
                     and text projections for classification.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    BlipForQuestionAnswering,
    CLIPModel,
)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ModelStrategy(ABC):
    """
    Abstract base class defining the interface that all VQA model strategies
    must implement.
    """

    @abstractmethod
    def load_model(self, num_classes: int, device: torch.device) -> None:
        """
        Load and configure the model for the given number of output classes.

        Parameters
        ----------
        num_classes : Total number of answer classes.
        device      : Target device (CPU or CUDA).
        """
        ...

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Perform a forward pass and return class logits.

        Parameters
        ----------
        batch : Dictionary produced by VQADataset containing
                pixel_values, input_ids, attention_mask, and answer_label.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes).
        """
        ...

    @abstractmethod
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Return predicted class indices for a batch (no gradient computation).

        Returns
        -------
        torch.Tensor of shape (batch_size,).
        """
        ...

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying nn.Module for optimizer or checkpoint access."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable identifier for this strategy."""
        ...


# ---------------------------------------------------------------------------
# BLIP strategy
# ---------------------------------------------------------------------------


class _BLIPClassificationModel(nn.Module):
    """
    Wraps the BLIP question-answering model and replaces the generative
    text decoder with a classification head.

    Architecture
    ------------
    1. BlipVisionModel encodes the image into patch embeddings.
    2. BlipTextModel encodes the question with cross-attention over the
       image embeddings, producing a joint visual-linguistic representation.
    3. The [CLS] token (position 0) is passed through a 3-layer MLP head:
       Linear(hidden, 512) -> LayerNorm -> GELU -> Dropout(dropout) -> Linear(512, C)

    Vision freeze
    -------------
    When freeze_vision=True the ViT encoder is frozen at init.
    Call unfreeze_vision() after N warm-up epochs.
    """

    def __init__(
        self, blip_model: BlipForQuestionAnswering, num_classes: int,
        freeze_vision: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.blip = blip_model
        if freeze_vision:
            self._set_vision_grad(False)
        # BLIP text encoder hidden size is 768 for the base variant
        hidden_size = blip_model.config.text_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def _set_vision_grad(self, requires_grad: bool) -> None:
        for p in self.blip.vision_model.parameters():
            p.requires_grad = requires_grad

    def unfreeze_vision(self) -> None:
        self._set_vision_grad(True)
        print("[BLIPModel] Vision encoder unfrozen.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Step 1: encode the image with ViT
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state  # (B, P+1, H_vis)

        # Step 2: create a full-attention mask for the image sequence
        image_attn_mask = torch.ones(
            image_embeds.size()[:2], dtype=torch.long, device=image_embeds.device
        )

        # Step 3: encode the question with cross-attention to image patches
        text_outputs = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn_mask,
            return_dict=True,
        )

        # Step 4: take the [CLS] token embedding and classify
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]  # (B, H_text)
        logits = self.classifier(cls_embedding)                   # (B, num_classes)
        return logits


class BLIPStrategy(ModelStrategy):
    """
    Concrete strategy backed by Salesforce/blip-vqa-base.

    The generative decoder is discarded; a 3-layer MLP head is trained on
    top of the multimodal encoder's [CLS] token output.
    """

    PRETRAINED_ID = "Salesforce/blip-vqa-base"

    def __init__(self, freeze_vision: bool = True, dropout: float = 0.2) -> None:
        self._model: Optional[_BLIPClassificationModel] = None
        self._device: Optional[torch.device] = None
        self._freeze_vision = freeze_vision
        self._dropout = dropout

    def unfreeze_vision(self) -> None:
        if self._model is not None:
            self._model.unfreeze_vision()

    def load_model(self, num_classes: int, device: torch.device) -> None:
        self._device = device
        print(f"[BLIPStrategy] Loading '{self.PRETRAINED_ID}' ...")
        blip_base = BlipForQuestionAnswering.from_pretrained(self.PRETRAINED_ID)
        freeze = getattr(self, "_freeze_vision", True)
        self._model = _BLIPClassificationModel(
            blip_base, num_classes, freeze_vision=freeze, dropout=self._dropout
        ).to(device)
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self._model.parameters())
        print(f"[BLIPStrategy] {trainable:,} trainable / {total:,} total params")

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Call load_model() before forward().")
        pixel_values = batch["pixel_values"].to(self._device)
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)
        return self._model(pixel_values, input_ids, attention_mask)

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(batch)
        return logits.argmax(dim=-1)

    def get_model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return self._model

    def get_name(self) -> str:
        return "BLIP"


# ---------------------------------------------------------------------------
# CLIP strategy
# ---------------------------------------------------------------------------


class _CLIPFusionModel(nn.Module):
    """
    Multimodal classification model built on CLIP.

    Architecture
    ------------
    1. CLIP image encoder projects the image to a D-dimensional vector.
    2. CLIP text encoder projects the question to a D-dimensional vector.
    3. The two projections are concatenated (2D dimensions) and passed
       through a two-layer MLP fusion head to produce class logits.

    The CLIP backbone is frozen by default; only the fusion head is trained.
    """

    def __init__(self, clip_model: CLIPModel, num_classes: int) -> None:
        super().__init__()
        self.clip = clip_model

        # Freeze the CLIP backbone to save GPU memory and speed up training
        for param in self.clip.parameters():
            param.requires_grad = False

        embed_dim = clip_model.config.projection_dim  # 512 for vit-base-patch32
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.clip.get_image_features(
            pixel_values=pixel_values
        )  # (B, D)
        text_embeds = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (B, D)
        fused = torch.cat([image_embeds, text_embeds], dim=-1)  # (B, 2D)
        logits = self.fusion_head(fused)                         # (B, num_classes)
        return logits


class CLIPStrategy(ModelStrategy):
    """
    Concrete strategy backed by openai/clip-vit-base-patch32.

    A lightweight MLP fusion head is trained on top of the frozen CLIP
    image and text projections.

    Note: When using CLIPStrategy, pass a CLIPProcessor (not BlipProcessor)
    to DatasetProcessor so that images are resized to 224x224 as CLIP expects.
    """

    PRETRAINED_ID = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        self._model: Optional[_CLIPFusionModel] = None
        self._device: Optional[torch.device] = None

    def load_model(self, num_classes: int, device: torch.device) -> None:
        self._device = device
        print(f"[CLIPStrategy] Loading '{self.PRETRAINED_ID}' ...")
        clip_base = CLIPModel.from_pretrained(self.PRETRAINED_ID)
        self._model = _CLIPFusionModel(clip_base, num_classes).to(device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Call load_model() before forward().")
        pixel_values = batch["pixel_values"].to(self._device)
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)
        return self._model(pixel_values, input_ids, attention_mask)

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(batch)
        return logits.argmax(dim=-1)

    def get_model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return self._model

    def get_name(self) -> str:
        return "CLIP"
