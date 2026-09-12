from pathlib import Path
import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1

# -----------------------------------
# Load FaceNet models
# -----------------------------------

resnet = InceptionResnetV1(pretrained='vggface2').eval()

mtcnn = MTCNN(
    image_size=240,
    keep_all=True,
    min_face_size=60
)


# -----------------------------------
# Face Registration Function
# -----------------------------------

def registration(name, adm, foldername):

    # Location of the person's images
    img_dir = Path('data', foldername)

    if not img_dir.exists():
        return 'folder does not exist'

    # -----------------------------------
    # Load images using ImageFolder
    # -----------------------------------

    dataset = datasets.ImageFolder(root=img_dir)

    idx_to_class = {
        idx: class_name
        for class_name, idx in dataset.class_to_idx.items()
    }

    # -----------------------------------
    # Return one image at a time
    # -----------------------------------

    def collate_fn(batch):
        return batch[0]

    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn
    )

    # Store embeddings for each person
    embeddings = {
        class_name: []
        for class_name in dataset.class_to_idx
    }

    # -----------------------------------
    # Process every image
    # -----------------------------------

    for image, label in loader:

        faces, probabilities = mtcnn(
            image,
            return_prob=True
        )

        # No face detected
        if faces is None:
            continue

        # More than one face detected
        if len(faces) != 1:
            return 'found more than one face'

        # Face detection confidence
        probability = probabilities[0]

        if probability < 0.7:
            continue

        # -----------------------------------
        # Generate FaceNet embedding
        # -----------------------------------

        with torch.no_grad():
            embedding = resnet(faces)

        class_name = idx_to_class[label]

        embeddings[class_name].append(
            embedding.squeeze(0)
        )

    # -----------------------------------
    # Make sure enough good images exist
    # -----------------------------------

    class_name = foldername

    if len(embeddings[class_name]) < 5:
        return 'not enough valid face images'

    # -----------------------------------
    # Calculate average embedding
    # -----------------------------------

    average_embeddings = {}

    for class_name, person_embeddings in embeddings.items():

        stacked_embeddings = torch.stack(person_embeddings)

        average_embedding = torch.mean(
            stacked_embeddings,
            dim=0
        )

        average_embeddings[class_name] = average_embedding

    # -----------------------------------
    # Prepare data for saving
    # -----------------------------------

    embeddings_to_save = [
        (embedding, name, adm)
        for embedding in average_embeddings.values()
    ]

    # -----------------------------------
    # Load existing registrations
    # -----------------------------------

    embeddings_file = Path('results/embeddings.pt')

    embeddings = torch.load(
        embeddings_file
    )

    # Add new registration
    embeddings.extend(embeddings_to_save)

    # Save updated registrations
    torch.save(
        embeddings,
        embeddings_file
    )

   
    return 'success'


def recognition(image_path, threshold=0.8):

    # -----------------------------------
    # Load image
    # -----------------------------------

    image = Image.open(image_path).convert('RGB')

    # -----------------------------------
    # Detect face
    # -----------------------------------

    faces, probabilities = mtcnn(
        image,
        return_prob=True
    )

    if faces is None:
        return 'No face detected'

    # We want exactly one face
    if len(faces) != 1:
        return 'Please provide an image with exactly one face'

    # Check face detection confidence
    if probabilities[0] < 0.7:
        return 'Face detection confidence is too low'

    # -----------------------------------
    # Create embedding for input face
    # -----------------------------------

    with torch.no_grad():
        query_embedding = resnet(faces)

    query_embedding = query_embedding.squeeze(0)

    # -----------------------------------
    # Load registered embeddings
    # -----------------------------------

    embeddings_file = Path('results/embeddings.pt')

    if not embeddings_file.exists():
        return 'No registered faces found'

    registered_embeddings = torch.load(
        embeddings_file,
        weights_only=False
    )

    if len(registered_embeddings) == 0:
        return 'No registered faces found'

    # -----------------------------------
    # Find closest registered face
    # -----------------------------------

    best_distance = float('inf')
    best_name = None
    best_adm = None

    for embedding, name, adm in registered_embeddings:

        distance = torch.norm(
            query_embedding - embedding
        ).item()

        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_adm = adm

    # -----------------------------------
    # Check threshold
    # -----------------------------------

    if best_distance <= threshold:

        return {
            'name': best_name,
            'admission_number': best_adm,
            'distance': best_distance
        }

    return {
        'name': 'Unknown',
        'admission_number': None,
        'distance': best_distance
    }




