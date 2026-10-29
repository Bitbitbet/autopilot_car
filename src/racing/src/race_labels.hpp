#pragma once

// 模型输出类别顺序：索引就是 PredictResult::type，必须与模型的 label_list.txt 一致。
enum : int {
    LABEL_CONE = 0,    // 锥桶
    LABEL_PERSON = 1,  // 行人
    LABEL_BUSY = 2,    // 施工区标
    LABEL_LIMIT = 3,   // 限速
    LABEL_UNLIMIT = 4, // 解除限速
    LABEL_STOP = 5,    // 禁止通行
    LABEL_FLOW = 6,    // 畅通
    LABEL_PARK = 7,    // 停车场
    LABEL_GATE = 8,    // 道闸
    LABEL_CROSS = 9,   // 斑马线
    LABEL_FORK = 10,   // 岔路标志
    LABEL_LEFT = 11,   // 左转标志
    LABEL_CHOICE = 12, // 双向箭头
    LABEL_YLEFT = 13,  // 左岔路口
    LABEL_YRIGHT = 14, // 右岔路口
};
