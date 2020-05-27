package com.leetcode;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 26/5/2020 9:40 AM
 * @Author: lihongbing
 */

//460. LFU缓存   Least Frequently Used ，最近最少使用算法
// 如果一个数据在最近一段时间很少被访问到，那么可以认为在将来它被访问的可能性也很小。因此，当空间满时，最小频率访问的数据最先被淘汰
//写法1：HashMap<Integer, Node> cache 存缓存的内容;
//      min 是最小访问频次;
//      HashMap<Integer, LinkedHashSet<Node>> freqMap 存每个访问频次对应的 Node 的双向链表
// （写法 1 为了方便，直接用了 JDK 现有的 LinkedHashSet，其实现了 1 条双向链表贯穿哈希表中的所有 Entry，
//   支持以插入的先后顺序对原本无序的 HashSet 进行迭代）
//
//时间复杂度：O(1)
//
public class LFUCache {
    Map<Integer, Node> cache;   // 存储缓存的内容  key -> Node(key,value)
    Map<Integer, LinkedHashSet<Node>> freqMap; // 存储每个频次对应的双向链表
    int size;
    int capacity;
    int min; // 存储当前最小频次

    public LFUCache(int capacity) {
        cache = new HashMap<>(capacity);
        freqMap = new HashMap<>();
        this.capacity = capacity;
    }

    public int get(int key) {
        Node node = cache.get(key);
        if (node == null) {
            return -1;
        }
        freqInc(node);
        return node.value;
    }

    public void put(int key, int value) {
        if (capacity == 0) {
            return;
        }
        Node node = cache.get(key);
        if (node != null) {
            node.value = value;
            freqInc(node);
        } else {
            if (size == capacity) {
                Node deadNode = removeNode();
                cache.remove(deadNode.key);
                size--;
            }
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            addNode(newNode);
            size++;
        }
    }

    void freqInc(Node node) {
        // 从原freq对应的链表里移除, 并更新min
        int freq = node.freq;
        LinkedHashSet<Node> set = freqMap.get(freq); //之前已经保证了set不为null
        set.remove(node);
        if (freq == min && set.size() == 0) {
            min = freq + 1;
        }
        // 加入新freq对应的链表
        node.freq++;
        LinkedHashSet<Node> newSet = freqMap.get(freq + 1);
        if (newSet == null) {
            newSet = new LinkedHashSet<>();
            freqMap.put(freq + 1, newSet);
        }
        newSet.add(node);
    }

    void addNode(Node node) {
        LinkedHashSet<Node> set = freqMap.get(1);
        if (set == null) {
            set = new LinkedHashSet<>();
            freqMap.put(1, set);
        }
        set.add(node);
        min = 1;
    }

    Node removeNode() {
        LinkedHashSet<Node> set = freqMap.get(min);
        Node deadNode = set.iterator().next(); //因为是LinkedHashSet（HashSet+双向链表），第一个元素就是最久被访问的。
        set.remove(deadNode);
        return deadNode;
    }


    class Node {
        int key;
        int value;
        int freq = 1;

        public Node() {
        }

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

}


