import numpy as np
import matplotlib.pyplot as plt


def estimate_rotation_and_cs(laser_edge, Resolution, RctangleCS_leftedge, image_raw):
    # Reshape the image dimensions
    Ysize, Xsize = image_raw.shape
    columnsInImage, rowsInImage = np.meshgrid(np.arange(1, Xsize + 1), np.arange(1, Ysize + 1))
    columnsInImage_mm = columnsInImage * Resolution
    rowsInImage_mm = rowsInImage * Resolution

    # Detect angle of Laser, for rotating laser to vertical
    leftedgeRectangle = np.logical_and(
        np.logical_and(
            rowsInImage >= RctangleCS_leftedge[0][0], rowsInImage <= RctangleCS_leftedge[0][1]
        ),
        np.logical_and(
            columnsInImage >= RctangleCS_leftedge[1][0], columnsInImage <= RctangleCS_leftedge[1][1]
        ),
    )
    left_edge = np.logical_and(laser_edge, leftedgeRectangle)

    # Check if left_edge is empty
    if not np.any(left_edge):
        raise ValueError(
            "left_edge is empty, cannot perform polynomial fitting. Consider Widening Left Rectange Range"
        )
    else:
        # print(f"length of left_edge = {len(left_edge)}")
        pass

    # Debug: Plot the left edge mask image data
    # plt.figure()
    # plt.imshow(left_edge, cmap='gray')
    # plt.title('left edge mask')
    # plt.xlabel('Column Pixel')
    # plt.ylabel('Row Pixel')
    # plt.colorbar()

    left_edge_y, left_edge_x = np.where(left_edge == 1)
    x_left = np.median(left_edge_x)
    leftedge_polyfit = np.polyfit(left_edge_y, left_edge_x, 1)
    leftedge_angle = np.degrees(np.arctan(leftedge_polyfit[0]))

    # Debug: Plot the left edge mask image data
    #   Generate x values for the fitted line
    # fit_x = np.linspace(min(left_edge_y), max(left_edge_y), 100)
    # fit_y = np.polyval(leftedge_polyfit, fit_x)
    # plt.figure()
    # plt.imshow(image_raw, cmap='gray')
    # plt.scatter(left_edge_x, left_edge_y, color='red', s=1, label='Detected Edges')
    # plt.plot(fit_y, fit_x, color='blue', label='Fitted Line')
    # plt.title('Left Edge Detection with Polyfit')
    # plt.xlabel('Column Pixel')
    # plt.ylabel('Row Pixel')
    # plt.legend()
    # plt.show()

    # Find CS of laser center, for shifting CS Origin to laser center
    indexy, indexx = np.where(laser_edge)
    x_mid = np.mean(indexx)
    y_mid = np.mean(indexy)
    center_CS = [x_mid * Resolution, (Ysize - y_mid + 1) * Resolution]

    # NOTE these coordinates and angles seem to differ vary from matlab values within +-0.5

    return leftedge_angle, center_CS
