package com.leetcode;

import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.TreeMap;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 19/6/2020 3:28 PM
 * @Author: lihongbing
 */
//分治
public class Divide_and_Conquer {
    //剑指 Offer 40. 最小的k个数
    //输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
    //
    //示例 1：
    //输入：arr = [3,2,1], k = 2
    //输出：[1,2] 或者 [2,1]
    //示例 2：
    //输入：arr = [0,1,2,1], k = 1
    //输出：[0]
    //方法一：排序
    //思路和算法
    //
    //对原数组从小到大排序后取出前 k 个数即可。
    //时间复杂度：O(nlogn)，其中 n 是数组 arr 的长度。算法的时间复杂度即排序的时间复杂度。
    //空间复杂度：O(logn)，排序所需额外的空间复杂度为 O(logn)。

    //方法二：堆
    //思路和算法
    //我们用一个大根堆实时维护数组的前 k 小值。首先将前 k 个数插入大根堆中，随后从第 k+1 个数开始遍历，
    // 如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数。
    // 最后将大根堆里的数存入数组返回即可。在下面的代码中，由于 C++ 语言中的堆（即优先队列）为大根堆，我们可以这么做。而 Python 语言中的对为小根堆，因此我们要对数组中所有的数取其相反数，才能使用小根堆维护前 k 小值。
    //时间复杂度：O(NlogK)
    public int[] getLeastNumbers(int[] arr, int k) {
        // PriorityQueue默认是小根堆（堆顶元素最小），实现大根堆需要重写一下比较器。
        Queue<Integer> priorityQueue = new PriorityQueue<>((v1, v2) -> v2 - v1);
        for (int elemnt : arr) {
            priorityQueue.add(elemnt);
            if (priorityQueue.size() > k) {
                priorityQueue.remove();
            }
        }
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = priorityQueue.poll();
        }
        return result;
    }

    //方法三、二叉搜索树也可以 O(NlogK)解决 TopK 问题哦
    //BST 相对于前两种方法没那么常见，但是也很简单，和大根堆的思路差不多～
    //要提的是，与前两种方法相比，BST 有一个好处是求得的前K大的数字是有序的。
    //
    //因为有重复的数字，所以用的是 TreeMap 而不是 TreeSet（有的语言的标准库自带 TreeMultiset，也是可以的）。
    //
    //TreeMap的key 是数字，value 是该数字的个数。
    //我们遍历数组中的数字，维护一个数字总个数为 K 的 TreeMap：
    //1.若目前 map 中数字个数小于 K，则将 map 中当前数字对应的个数 +1；
    //2.否则，判断当前数字与 map 中最大的数字的大小关系：若当前数字大于等于 map 中的最大数字，就直接跳过该数字；若当前数字小于 map 中的最大数字，则将 map 中当前数字对应的个数 +1，并将 map 中最大数字对应的个数减 1。
    public int[] getLeastNumbers2(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // TreeMap的key是数字, value是该数字的个数。
        // cnt表示当前map总共存了多少个数字。
        TreeMap<Integer, Integer> map = new TreeMap<>();
        int cnt = 0;
        for (int num : arr) {
            // 1. 遍历数组，若当前map中的数字个数小于k，则map中当前数字对应个数+1
            if (cnt < k) {
                map.put(num, map.getOrDefault(num, 0) + 1);
                cnt++;
                continue;
            }
            // 2. 否则，取出map中最大的Key（即最大的数字), 判断当前数字与map中最大数字的大小关系：
            //    若当前数字比map中最大的数字还大，就直接忽略；
            //    若当前数字比map中最大的数字小，则将当前数字加入map中，并将map中的最大数字的个数-1。
            Map.Entry<Integer, Integer> entry = map.lastEntry(); //返回TreeMap中key最大的键值对
            if (entry.getKey() > num) {
                map.put(num, map.getOrDefault(num, 0) + 1);
                if (entry.getValue() == 1) {
                    map.pollLastEntry(); //删除TreeMap中key最大的键值对
                } else {
                    map.put(entry.getKey(), entry.getValue() - 1);
                }
            }

        }

        // 最后返回map中的元素
        int[] res = new int[k];
        int idx = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int freq = entry.getValue();
            while (freq-- > 0) {
                res[idx++] = entry.getKey();
            }
        }
        return res;
    }
}
