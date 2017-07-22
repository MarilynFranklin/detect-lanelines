# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:

1. Images are converted to grayscale
      ```
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      ```

      ![](reflection_images/gray.jpg?raw=true)

2. Next, I blurred the image with Gausian smoothing to remove noise
      ```
      blur = cv2.GaussianBlur(gray, (5, 5), 0)
      ```

      ![](reflection_images/blurry.jpg?raw=true)

3. Then, I applied Canny edge detection with a low threshold of 50 and a high
   threshold of 150. Canny works by detecting strong edges that have a pixel gradient
   above the high threshold. It rejects pixel gradients that are below the
   low threshold. Then, it keeps pixel gradients that are between the
   low and high thresholds and close to the strong edges it detected.
      ```
      edges = cv2.Canny(blur, 50, 150)
      ```

      ![](reflection_images/edges.jpg?raw=true)

4. Next, I select a trapezoid around the bottom of the image where we expected
   to find lanes.
      ```
      imshape = image.shape
      third = int(imshape[0] * .66)
      half = int(imshape[1] * .50)
      vertices = np.array(
          [[(0,imshape[0]),
            (half - 130, third),
            (half + 130, third),
            (imshape[1],imshape[0])]],
          dtype=np.int32)
      masked_edges = region_of_interest(edges, vertices)
      ```

      ![](reflection_images/masked_edges.jpg?raw=true)

5. Finally, I used opencv's `HoughLinesP` function to find lines with of a
   minimum length
      ```
      rho = 3           # distance resolution in pixels of the Hough grid
      theta = np.pi/180 # angular resolution in radians of the Hough grid
      threshold = 15    # minimum number of votes (intersections in Hough grid cell)
      min_line_len =  5 # minimum number of pixels making up a line
      max_line_gap = 25 # maximum gap in pixels between connectable line segments

      lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
      )
      ```

      ![](reflection_images/with_lines.jpg?raw=true)

#### Updates to the `draw_lines` function

In order to draw a single line on the left and right lanes, I updated the
draw_lines function to filter lines by slope and by their x coordinate. Points
with an x coordinate less than half of the width of the image were used for the
left lane, and the other points were used for the right lane. Those points were
then fit to a linear model using `RANSACRegressor` from `sklearn`. `RANSACRegressor` removes outliers which helped to fit some of the lines in the challenge video to the correct lane. Then, I used the `coef_` (slope) and `intercept_` (y intercept) from the linear model to find the x coordinates at the bottom of the image and in the middle of the image. These coordinates were used to plot a single line for the left and right lanes.

```
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    x_right = []
    y_right = []
    x_left = []
    y_left = []

    halfway = int(image.shape[1] * .50)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2 - x1) == 0:
                continue
            slope = ((y2-y1)/(x2-x1))

            if x1 <= halfway and x2 <= halfway and slope < -0.5:
                x_left.extend(([x1], [x2]))
                y_left.extend(([y1], [y2]))
            elif x1 > halfway and x2 > halfway and slope > 0.5:
                x_right.extend(([x1], [x2]))
                y_right.extend(([y1], [y2]))

    minright_y = minleft_y = 360
    maxleft_y  = maxright_y = 720

    if len(x_left) >= 2 and len(y_left) >=2:
        left_fit = RANSACRegressor(LinearRegression(), residual_threshold=10.0)
        left_fit.fit(x_left, y_left)

        # Left Line
        m_left = left_fit.estimator_.coef_
        b_left = left_fit.estimator_.intercept_

        if m_left != 0:
            minleft_x = int((minleft_y - b_left)/m_left)
            maxleft_x = int((maxleft_y - b_left)/m_left)

            cv2.line(img, (minleft_x, minleft_y), (maxleft_x, maxleft_y), color,thickness)

    if len(x_right) >= 2 and len(y_right) >=2:
        right_fit = RANSACRegressor(LinearRegression(), residual_threshold=10.0)
        right_fit.fit(x_right, y_right)

        # Right Line
        m_right = right_fit.estimator_.coef_
        b_right = right_fit.estimator_.intercept_

        if m_right != 0:
            minright_x = int((minright_y - b_right)/m_right)
            maxright_x = int((maxright_y - b_right)/m_right)

            cv2.line(img, (minright_x, minright_y), (maxright_x, maxright_y), color,thickness)
```

      ![](reflection_images/image_with_lanes.jpg?raw=true)

### 2. Identify potential shortcomings with your current pipeline

One shortcoming visible in the challenge video is that shaddows make it difficult for my pipeline to correctly
detect lane lines.

Other potential shortcomings are night, rain, and snow.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to try normalizing colors before converting
grayscale. This may help issues caused by shaddows.
