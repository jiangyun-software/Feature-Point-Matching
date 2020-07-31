import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import shutil
import imutils
import argparse
import time
import random
import math
import glob

# 6H：最小match是4个
# 1-5H：最小match是10个
MIN_MATCH_COUNT = 10
count_num = 0
save_path = "/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_Affine_Results"
save_path_crop = "/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_Crop_Results"
save_path_cropresize = "/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_CropResize_Results"
# print("123")
# 1H img1: right
# 2H img1: right
# 3H img1: right_3H
# 4H img1: right
# 5H img1: right
# 6H img1: right_6H

for filename in glob.glob("/tf/SUANFAZU/detect_images/BT0981优化图片/0H/*.png"):
    print("filename: %d",filename)
    # if count_num == 0:
    #     template_image = cv2.imread("C:\\Jiangyun\\1H\\" + filename, 0)
    #     template_image = template_image[229:417, 876:1097]
    #     cv2.imwrite("C:\\Jiangyun\\detect_result\\template_1H.png", template_image)
    #     pass
    
    detect_image = cv2.imread(filename, 0)
    
    # template: 
    # Template_1H_new
    # Template_2H_new
    # Template_3H
    # Template_4H
    # Template_5H
    # Template_6H
    
    
    template_image = cv2.imread("/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_Template/Template_0H.png", 0)
    #img2 = cv2.imread("F:\\Semantic-Segmentation-Suite-master\\DeepLearningTools\\AOI\\4H\\" + filename, 0)
    count_num = count_num + 1
    # 使用SIFT检测角点
    sift = cv2.xfeatures2d.SIFT_create()
    # sift=cv2.SIFT()

    # 获取关键点和描述符
    kp1, des1 = sift.detectAndCompute(template_image, None)
    kp2, des2 = sift.detectAndCompute(detect_image, None)

    # 定义FLANN匹配器
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 使用KNN算法匹配
    matches = flann.knnMatch(des1, des2, k=2)
    

    # 去除错误匹配
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # 单应性
    if len(good) > MIN_MATCH_COUNT:
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        src_pts_old = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts_list = []
        #转换坐标系为左上角坐标系
        for pnt in src_pts:
            src_pts_list.append(((pnt[0] + 900, pnt[1] + 227))) #900, 227

        src_pts = np.asarray(src_pts_list)
        src_pts = src_pts.reshape(-1, 1, 2)
        
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # src_pts_new_A = src_pts_old[4]
        # src_pts_new_B = src_pts_old[14]
        # src_pts_new_C = src_pts_old[24]
        
        # dst_pts_new_A = dst_pts[4]
        # dst_pts_new_B = dst_pts[14]
        # dst_pts_new_C = dst_pts[24]

        # src_pts_new = np.float32([src_pts_old[q] for q in range(3)])
        # dst_pts_new = np.float32([dst_pts[t] for t in range(3)])

        


        # src_pts_new = np.float32([src_pts_new_A, src_pts_new_B, src_pts_new_C])
        # dst_pts_new = np.float32([dst_pts_new_A, dst_pts_new_B, dst_pts_new_C])
                
        # print("src_pts_new: ", src_pts_new[0][0][0])
        # print("dst_pts_new: ", dst_pts_new[0][0][0])
        # print("dst_pts Length: ", len(dst_pts))
        
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ravel方法将数据降维处理，最后并转换成列表格式
        matchesMask = mask.ravel().tolist()

        src_pts_new_x = []
        src_pts_new_y = []
        dst_pts_new_x = []
        dst_pts_new_y = []
        count_pts = 0
        for k in range(len(src_pts)):
            if matchesMask[k]==1:
                count_pts = count_pts + 1
                src_pts_new_x.append(src_pts_old[k][0][0])
                src_pts_new_y.append(src_pts_old[k][0][1])
                dst_pts_new_x.append(dst_pts[k][0][0])
                dst_pts_new_y.append(dst_pts[k][0][1])
        
        start = time.time()
        # 三角形面积最大
        s=0
        for i in range(count_pts-2):#x1不用遍历到最后一个数字
            for j in range(i+1,count_pts-1):
                s1=src_pts_new_x[j]-src_pts_new_x[i]
                s2=src_pts_new_y[j]-src_pts_new_y[i]#先求出x2-x1和y2-y1，避免后面重复计算
                for k in range(j+1,count_pts):
                    s4=abs(s1*(src_pts_new_y[k]-src_pts_new_y[i])-(src_pts_new_x[k]-src_pts_new_x[i])*s2)/2
                    if s<s4:
                        s=s4
                        idx_0 = i
                        idx_1 = j
                        idx_2 = k
        
        end = time.time()
        print("time: " + str(end - start))
        src_pts_new = np.float32([[src_pts_new_x[idx_0],src_pts_new_y[idx_0]], [src_pts_new_x[idx_1],src_pts_new_y[idx_1]], [src_pts_new_x[idx_2],src_pts_new_y[idx_2]]])
        dst_pts_new = np.float32([[dst_pts_new_x[idx_0],dst_pts_new_y[idx_0]], [dst_pts_new_x[idx_1],dst_pts_new_y[idx_1]], [dst_pts_new_x[idx_2],dst_pts_new_y[idx_2]]])

        print("src_pts_new: ", src_pts_new)
        print("dst_pts_new: ", dst_pts_new)
        M_new = cv2.getAffineTransform(dst_pts_new, src_pts_new)

        affined_image = cv2.warpAffine(detect_image, M_new, (detect_image.shape[1], detect_image.shape[0]))
        affined_image_bb = cv2.rectangle(affined_image, (203, 211), (1184, 432), (255,255,255), 2)
        #0H: (203, 211), (1184, 432)
        #1H: (303, 193), (1146, 420)
        #2H: (276, 172), (1413, 489)
        #3H: (292, 147),  (1020, 323)
        #4H: (172, 164)， (987, 379)
        #5H: (270, 221)， (916, 397)
        #6H: (245, 131), (982, 289)
        affined_image_crop = affined_image_bb[211:432, 203:1184] #2H: (303, 191), (1383, 470)
        
        affined_image_crop_resize = cv2.resize(affined_image_crop, (992, 224))
        
        cv2.imwrite(os.path.join(save_path, filename+'result.png'), affined_image_bb)   
        cv2.imwrite(os.path.join(save_path_crop, filename+'cropresult.png'), affined_image_crop) 
        cv2.imwrite(os.path.join(save_path_cropresize, filename+'cropresizeresult.png'), affined_image_crop_resize)
        
        # print("matchesMask Length: ", len(matchesMask))
        # 获取template_image的图像尺寸
        # h2, w2 = template_image.shape
        # pts2是图像img2的四个顶点
        # pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        # 计算变换后的四个顶点坐标位置
        # dst = cv2.perspectiveTransform(pts, M)
        # affined_image = cv2.warpAffine(detect_image, M, (detect_image.shape[1], detect_image.shape[0]))
        # transformed_image = cv2.warpPerspective(detect_image, M, (detect_image.shape[1], detect_image.shape[0]))
        # # 计算角度
        # cv2.imwrite("C:\\Jiangyun\\detect_result\\" + filename, transformed_image)
        # cv2.imwrite("F:\\Semantic-Segmentation-Suite-master\\DeepLearningTools\\detect_result\\" + filename, transformed_image[192:395, 180:980])
        # 获取img1的图像尺寸
        #h, w = template_image.shape
        # pts是图像img1的四个顶点
        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 计算变换后的四个顶点坐标位置
        #dst = cv2.perspectiveTransform(pts, M)

        # 根据四个顶点坐标位置在img2图像画出变换后的边框
        #img3 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # start_point = (170, 160)
    # end_point = (1000, 420)
    # color = (255, 255, 255)
    # thickness = 3
    # img3 = cv2.rectangle(img3, start_point, end_point, color, thickness)
    # img0 = cv2.rectangle(img0, start_point, end_point, color, thickness)
    #
    # detect_image = cv2.circle(detect_image,(src_pts_new[0][0][0],src_pts_new[0][0][1]),30,(255,255,255),-1)
    
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(template_image, kp1, detect_image, kp2, good, None, **draw_params)
    if len(good) > MIN_MATCH_COUNT:
        # for k in range(len(dst_pts)):
        #     if matchesMask[k]==1:
        #         dst_pts_new_x = dst_pts[k][0][0] + np.float32(1280)
        #         dst_pts_new_y = dst_pts[k][0][1]
        for k in range(len(dst_pts_new)):
            img4 = cv2.circle(img3,(dst_pts_new[k][0]+ np.float32(1280),dst_pts_new[k][1]),8,(255,255,255),-1)
            img5 = cv2.circle(img3,(src_pts_new[k][0],src_pts_new[k][1]),8,(0,0,255),-1)
        plt.imshow(img5,'gray')
        plt.show()
#         cv2.imwrite("C:\\Jiangyun\\matching_result\\" + filename, img5)
#     else:
#         cv2.imwrite("C:\\Jiangyun\\matching_result\\" + filename, img3)
   
    # crop_img = img0[160:420, 170:1000]
    # crop_img2 = img2[180:395, 200:980]
    # crop_img3 = img3[190:410, 350:1700]
    # print('Img Count_num = %d' % Count_num)
    # print('img2 original: ', pts2)
    # print('img2 transformed: ', dst2)
    # print('img1 original: ', pts)
    # print('img1 transformed: ', dst)
    # plt.imshow(img0)
    # plt.show()
    # plt.imshow(affined_image, 'gray')
    # plt.show()
    # plt.imshow(crop_img)
    # plt.show()
    # plt.imshow(crop_img2)
    # plt.show()
    # plt.imshow(crop_img3)
    # plt.show()
    # plt.waitKey(0)
    # "cv2.imshow" may lead to a Kernel crash so we change it to the 'plt' handle to check the image
    # cv2.imshow("Image show", img3)
    # cv2.waitKey(0)
    # cv2.imwrite(os.path.join(save_path, filename + 'result.png'), img3)

print("for loop complete!!!!!!!!!!!")

for result_file in glob.glob("/tf/SUANFAZU/detect_images/BT0981优化图片/0H/*.pngresult.png"): # move the result images to the "results" folder
    shutil.move(result_file,"/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_Affine_Results")
    
for result_file in glob.glob("/tf/SUANFAZU/detect_images/BT0981优化图片/0H/*.pngcropresult.png"): # move the result images to the "results" folder
    shutil.move(result_file,"/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_Crop_Results")
    
for result_file in glob.glob("/tf/SUANFAZU/detect_images/BT0981优化图片/0H/*.cropresizeresult.png"): # move the result images to the "results" folder
    shutil.move(result_file,"/tf/SUANFAZU/detect_images/BT0981优化图片/0H/0H_CropResize_Results")
