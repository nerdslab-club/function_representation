import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class SiameseNetwork(pl.LightningModule):
    def __init__(self, embedding_size=768):
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.encoder = nn.Sequential(
            nn.Linear(300 * embedding_size, 3 * embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(3 * embedding_size, embedding_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.encoder(x)

    def triplet_loss(self, anchor, positive, negative, margin=0.6):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.relu(distance_positive - distance_negative + margin))
        return loss

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_embedding = self(anchor)
        positive_embedding = self(positive)
        negative_embedding = self(negative)

        loss = self.triplet_loss(
            anchor_embedding, positive_embedding, negative_embedding
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        print(batch_idx, loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def init_network(self, init_weights=True):
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)

    def save_network(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def resume_training(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

    def similarity_inference(self, input1, input2):
        embedding1 = self(input1)
        embedding2 = self(input2)
        similarity_score = F.pairwise_distance(embedding1, embedding2)
        return similarity_score

    def getSiameseEmbedding(self, input1):
        return self(input1)
