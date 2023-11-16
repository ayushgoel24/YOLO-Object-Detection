import torch

class YOLOLoss:

    @staticmethod
    def compute_loss(output, target, lambda_coordinate=5, lambda_non_objectness=0.5):
        """
        Compute the YOLO loss between the model output and the target.
        Args:
            output: Tensor, predicted values [B, 8, 8, 8]
            target: Tensor, ground truth values [B, 8, 8, 8]
            lambda_coordinate: Weight for coordinate loss.
            lambda_non_objectness: Weight for no-object loss.
        Returns:
            Scalar tensor, the computed loss
        """

        # Find indices of cells with and without objects
        object_indices = torch.nonzero(target[:, 0, :, ] != 0)
        no_object_indices = torch.nonzero(target[:, 0, :, ] == 0)

        # Filter outputs and targets based on object presence
        target_objects = target[object_indices[:, 0], :, object_indices[:, 1], object_indices[:, 2]]
        output_objects = output[object_indices[:, 0], :, object_indices[:, 1], object_indices[:, 2]]

        # Compute the difference for location predictions
        location_loss_xy = torch.sum((target_objects[:, 1:3] - output_objects[:, 1:3]) ** 2)
        location_loss_wh = torch.sum((torch.sqrt(target_objects[:, 3:5]) - torch.sqrt(output_objects[:, 3:5])) ** 2)
        location_loss = lambda_coordinate * (location_loss_xy + location_loss_wh)

        # Compute classification loss
        classification_loss = torch.sum((target_objects[:, 5:] - output_objects[:, 5:]) ** 2)

        # Compute Intersection over Union (IoU) for objectness score calculation
        # This assumes an external function called IoU_calc
        predicted_box = get_box_coordinates(output_objects, object_indices)
        target_box = get_box_coordinates(target_objects, object_indices)
        iou = IoU_calc(predicted_box, target_box)

        # Compute objectness losses
        objectness_loss = torch.sum(target_objects[:, 0] * (iou - output_objects[:, 0]) ** 2)
        no_object_targets = target[no_object_indices[:, 0], :, no_object_indices[:, 1], no_object_indices[:, 2]]
        no_object_outputs = output[no_object_indices[:, 0], :, no_object_indices[:, 1], no_object_indices[:, 2]]
        no_object_loss = lambda_non_objectness * torch.sum(no_object_targets[:, 0] * (0 - no_object_outputs[:, 0]) ** 2)

        confidence_loss = objectness_loss + no_object_loss

        # Total loss
        loss = location_loss + confidence_loss + classification_loss

        return loss
