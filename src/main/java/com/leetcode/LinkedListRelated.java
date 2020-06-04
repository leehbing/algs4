package com.leetcode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 3/6/2020 5:14 PM
 * @Author: lihongbing
 */
//链表相关的
public class LinkedListRelated {
    //206. 反转链表
    public static ListNode reverseList(ListNode head) {
        //方法一：增加两个指针，遍历的时候保留临时指针
        //时间复杂度：O(n)，假设 n 是列表的长度，时间复杂度是 O(n)。
        //空间复杂度：O(1)
        ListNode pre = null;
        ListNode next = head;
        while (next != null) {
            ListNode temp = next.next; //必须
            next.next = pre;
            pre = next;
            next = temp;
        }
        return pre;

    }

    //206. 反转链表
    public ListNode reverseList2(ListNode head) {
        //方法二：递归思想：https://leetcode-cn.com/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-by-leetcode/
        //递归比较难以理解，看评论：看了半个小时可算是把这个递归看懂了！
        // 不妨假设链表为1，2，3，4，5。按照递归，当执行reverseList（5）的时候返回了5这个节点，reverseList(4)中的p就是5这个节点，
        // 我们看看reverseList（4）接下来执行完之后，5->next = 4, 4->next = null。这时候返回了p这个节点，也就是链表5->4->null，
        // 接下来执行reverseList（3），代码解析为4->next = 3,3->next = null，这个时候p就变成了，5->4->3->null, reverseList(2),
        // reverseList(1)依次类推，p就是:5->4->3->2->1->null
        //时间复杂度：O(n)，假设 nn 是列表的长度，那么时间复杂度为 O(n)。
        //空间复杂度：O(n)，由于使用递归，将会使用隐式栈空间。递归深度可能会达到 n 层。
        if (head == null || head.next == null) return head;
        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    //92. 反转链表 II      反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
    public static ListNode reverseBetween(ListNode head, int m, int n) {
        //思想：计数器，将要反转的部分截取出来反转，然后在拼回去
        //比如输入 1 -> 2 -> 3 -> 4 -> 5 -> null     2,4
        //        n1   p         q    n2
        //首先扫描一遍链表，利用p,q记录m和n对应的位置，n1和n2记录m-1和n+1的位置
        //然后将q.next=null，针对p这个子链表进行反转，最后再拼接回去
        //时间复杂度：第一次会扫描一遍整个链表，然后会扫描一遍子链表[m,n] ，O(l+n-m),其实达不到题目所要求的只扫描一遍的要求。
        int i = 0;
        ListNode cur = head;
        ListNode p = head;
        ListNode q = null;

        ListNode n1 = null;
        ListNode n2 = null;

        while (cur != null) {
            i++;
            if (i == m - 1) n1 = cur;
            if (i == m) p = cur; //肯定不为空
            if (i == n) q = cur; //肯定不为空
            if (i == n + 1) n2 = cur;
            cur = cur.next;
        }


        q.next = null;
        //将p反转，同题目206
        ListNode pre = null;
        ListNode next = p;

        while (next != null) {
            ListNode temp = next.next;
            next.next = pre;
            pre = next;
            next = temp;
        }

        if (n1 != null) {
            n1.next = pre;
        } else {
            head = pre;
        }
        p.next = n2;


        return head;
    }
    ////思路：head表示需要反转的头节点，pre表示需要反转头节点的前驱节点
    // 同时我们也需要设置一个哑节点dummy，因为m=1时，我们可以也有前驱节点，剩余部分看代码即可。
    //    //我们需要反转n-m次，我们将head的next节点移动到需要反转链表部分的首部，需要反转链表部分剩余节点依旧保持相对顺序即可
    //反转的过程中，有点像头插法。
    //    //比如1->2->3->4->5,m=1,n=5
    //    //第一次反转：1(head) 2(next) 3 4 5 反转为 2 1 3 4 5
    //    //第二次反转：2 1(head) 3(next) 4 5 反转为 3 2 1 4 5
    //    //第三次发转：3 2 1(head) 4(next) 5 反转为 4 3 2 1 5
    //    //第四次反转：4 3 2 1(head) 5(next) 反转为 5 4 3 2 1
    //    ListNode* reverseBetween(ListNode* head, int m, int n) {
    //        ListNode *dummy=new ListNode(-1);
    //        dummy->next=head;
    //        ListNode *pre=dummy;
    //        for(int i=1;i<m;++i)pre=pre->next;
    //        head=pre->next;
    //        for(int i=m;i<n;++i){
    //            ListNode *nxt=head->next;
    //            //head节点连接nxt节点之后链表部分，也就是向后移动一位
    //            head->next=nxt->next;
    //            //nxt节点移动到需要反转链表部分的首部
    //            nxt->next=pre->next;
    //            pre->next=nxt;//pre继续为需要反转头节点的前驱节点
    //        }
    //        return dummy->next;
    //    }

    //234. 回文链表    请判断一个链表是否为回文链表。要求空间复杂度达到O(1)
    //输入: 1->2->2->1
    //输出: true
    //方法一，最简单的方法是，将值复制到数组中后用双指针法
    //时间复杂度：O(n)
    //空间复杂度：O(n)
    //方法二，递归，空间复杂度还是O(n)，太复杂了，可以看官网
    //方法三，将链表的后半部分反转（修改链表结构），然后将前半部分和后半部分进行比较。比较完成后我们应该将链表恢复原样。
    // 虽然不需要恢复也能通过测试用例，因为使用该函数的人不希望链表结构被更改。
    //          空间复杂度O(1)
    public static boolean isPalindrome(ListNode head) {
        if (head == null) return true;

        // Find the end of first half and reverse second half.
        ListNode firstHalfEnd = Utils.endOfFirstHalf(head); //通过快慢指针法找到前半部分的尾节点
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);

        // Check whether or not there is a palindrome.
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) result = false;
            p1 = p1.next;
            p2 = p2.next;
        }

        // Restore the list and return the result.
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }
    //160. 相交链表
    //两个链表是相交的，求相交的结点
    //方法一：暴力法
    //针对headA中的每个结点，查看在headB中是否存在，当存在就返回这个结点，这就是相交的结点
    //时间复杂度：O(mn)
    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = null;
        while (p1 != null) {
            p2 = headB;
            while (p2 != null) {
                if (p1 == p2) {
                    return p1;
                }
                p2 = p2.next;
            }
            p1 = p1.next;
        }
        return null;
    }

    //方法二：hashMap
    //时间复杂度 : O(m+n)
    //空间复杂度 : O(m) 或 O(n)
    public static ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = headB;
        Map<ListNode, Integer> map = new HashMap<>();
        while (p1 != null) {
            map.put(p1, 1);
            p1 = p1.next;
        }
        while (p2 != null) {
            if (map.containsKey(p2)) return p2;
            p2 = p2.next;
        }
        return null;
    }
    //方法三：双指针法，官方还有一种解法




    //21. 合并两个有序链表
    //将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode p = dummyHead;
        int cur = 0;
        while (l1 != null || l2 != null) {
            if (l1 == null) {
                cur = l2.val;
                l2 = l2.next;
            } else if (l2 == null) {
                cur = l1.val;
                l1 = l1.next;
            } else if (l1.val < l2.val) {
                cur = l1.val;
                l1 = l1.next;
            } else if (l1.val >= l2.val) {
                cur = l2.val;
                l2 = l2.next;
            }
            p.next = new ListNode(cur);
            p = p.next;
        }
        return dummyHead.next;
    }

    //88. 合并两个有序数组
    //给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
    //方法一：合并后排序
    //时间复杂度 : O((n+m)log(n+m))
    //空间复杂度 : O(1)
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        System.arraycopy(nums2, 0, nums1, m, n);
        Arrays.sort(nums1);
    }

    //方法二：双指针 / 从前往后
    //时间复杂度 : O(n + m)
    //空间复杂度 : O(m)
    public void merge2(int[] nums1, int m, int[] nums2, int n) {
        // Make a copy of nums1.
        int[] nums1_copy = new int[m];
        System.arraycopy(nums1, 0, nums1_copy, 0, m);

        // Two get pointers for nums1_copy and nums2.
        int p1 = 0;
        int p2 = 0;

        // Set pointer for nums1
        int p = 0;

        // Compare elements from nums1_copy and nums2
        // and add the smallest one into nums1.
        while ((p1 < m) && (p2 < n))
            nums1[p++] = (nums1_copy[p1] < nums2[p2]) ? nums1_copy[p1++] : nums2[p2++];

        // if there are still elements to add
        if (p1 < m)
            System.arraycopy(nums1_copy, p1, nums1, p1 + p2, m + n - p1 - p2);
        if (p2 < n)
            System.arraycopy(nums2, p2, nums1, p1 + p2, m + n - p1 - p2);
    }
}
