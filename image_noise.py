import numpy as np


class ImageNoise:
    def __init__(self, noise_type=None, strength=0.0):
        #Speichert die Rauschart und begrenzt ihre Stärke auf den Bereich von 0 bis 1
        self.noise_type = noise_type
        self.strength = np.clip(strength, 0.0, 1.0)
        self.rng = np.random.default_rng()

    def apply(self, image):
        #Gibt das Bild unverändert zurück, wenn kein Rauschen eingeschaltet ist
        if self.noise_type is None or self.strength == 0:
            return image

        image_float = image.astype(np.float32)

        #Addiert normalverteiltes Helligkeitsrauschen auf alle Bildpunkte
        if self.noise_type == "gaussian":
            noise = self.rng.normal(0, 50 * self.strength, image.shape)
            return np.clip(image_float + noise, 0, 255).astype(np.uint8)

        #Färbt zufällig ausgewählte Bildpunkte vollständig schwarz oder weiß
        if self.noise_type == "salt_pepper":
            result = image.copy()
            selection = self.rng.random(image.shape[:2])
            result[selection < 0.05 * self.strength] = 0
            result[selection > 1 - 0.05 * self.strength] = 255
            return result

        #Verstärkt das Rauschen abhängig von der ursprünglichen Helligkeit
        if self.noise_type == "speckle":
            noise = self.rng.normal(0, 0.5 * self.strength, image.shape)
            return np.clip(image_float + image_float * noise, 0, 255).astype(np.uint8)

        raise ValueError(f"Unbekannte Rauschart: {self.noise_type}")
