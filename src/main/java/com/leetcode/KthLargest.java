package com.leetcode;

import java.util.PriorityQueue;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 9/5/2020 11:04 AM
 * @Author: lihongbing
 */
//703. 数据流中的第K大元素
//思想，使用java的优先队列
//优先队列的容量为k，最顶的元素就是我们要的第k大的元素，如果新加的元素小于等于peek(),忽略，否则移除最顶端元素，加进新元素，再返回最顶端元素
class KthLargest {
    PriorityQueue<Integer> priorityQueue = null;
    int k;
    public KthLargest(int k, int[] nums) {
        this.k=k;
        priorityQueue = new PriorityQueue<>(k);
        for (int num : nums) {
            priorityQueue.add(num);
            if(priorityQueue.size()>k){
                priorityQueue.remove();
            }
        }
    }

    public int add(int val) {
        if(priorityQueue.size() < k) {
            priorityQueue.offer(val);

        }
        if(val > priorityQueue.peek()){
            priorityQueue.remove();
            priorityQueue.add(val);
        }
        return priorityQueue.peek();
    }
}