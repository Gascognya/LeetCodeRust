use std::collections::{HashMap, HashSet};
pub mod datastructure;
use datastructure::*;
pub struct Solution;

impl Solution {
    pub fn num_identical_pairs(nums: Vec<i32>) -> i32 {
        let mut count = 0;
        let len = nums.len();
        for index in 0..len {
            for i in index + 1..len {
                if nums[i] == nums[index] {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn kids_with_candies(candies: Vec<i32>, extra_candies: i32) -> Vec<bool> {
        let kid_num = candies.len();
        let mut res: Vec<bool> = vec![true; candies.len()];
        for kid in 0..kid_num {
            let candie_num = candies[kid] + extra_candies;
            for other in 0..kid_num {
                if candie_num < candies[other] {
                    res[kid] = false;
                    break;
                }
            }
        }
        res
    }

    pub fn shuffle(nums: Vec<i32>, n: i32) -> Vec<i32> {
        let mut res: Vec<i32> = Vec::new();
        for i in 0..n {
            res.push(nums[i as usize]);
            res.push(nums[(i + n) as usize]);
        }
        res
    }

    pub fn num_jewels_in_stones(j: String, s: String) -> i32 {
        let set = j.chars().collect::<HashSet<char>>();
        s.chars().filter(|x| set.contains(x)).count() as i32
    }

    pub fn reverse_left_words(s: String, n: i32) -> String {
        let mut res = String::new();
        res.push_str(&s[n as usize..]);
        res.push_str(&s[0..n as usize]);
        res
    }

    pub fn xor_operation(mut n: i32, start: i32) -> i32 {
        let mut i = 0;
        let mut res = 0;
        while n > 0 {
            res ^= start + 2 * i;
            i += 1;
            n -= 1;
        }
        res
    }

    pub fn defang_i_paddr(address: String) -> String {
        address.replace(".", "[.]")
    }

    pub fn min_count(coins: Vec<i32>) -> i32 {
        // let mut count = 0;
        // coins.iter().for_each(|n| count += (n+1)/2);
        // count
        coins.iter().map(|n| (n + 1) / 2).sum()
    }

    pub fn subtract_product_and_sum(n: i32) -> i32 {
        let cmp = n.to_string();
        let iter = cmp.chars().map(|c| c.to_digit(10).unwrap() as i32);
        iter.clone().fold(1, |acc, x| acc * x) - iter.clone().fold(0, |acc, x| acc + x)
    }

    pub fn smaller_numbers_than_current(nums: Vec<i32>) -> Vec<i32> {
        let mut m = HashMap::new();
        let mut v = Vec::new();
        for num in nums.iter() {
            if let Some(cache) = m.get(num) {
                v.push(*cache)
            } else {
                let mut count = 0;
                for n in nums.iter() {
                    if num > n {
                        count += 1;
                    }
                }
                m.insert(num, count);
                v.push(count)
            }
        }
        v
    }
    
    pub fn decompress_rl_elist(nums: Vec<i32>) -> Vec<i32> {
        let mut index = 0;
        let mut res = vec![];
        while index < nums.len() {
            res.extend(vec![nums[index + 1]; nums[index] as usize].iter());
            index += 2;
        }
        res
    }
    
    pub fn max_depth(s: String) -> i32 {
        let mut depth = 0;
        let mut max_depth = 0;
        let v = s.chars().collect::<Vec<char>>();
        for i in v.iter() {
            match i {
                '(' => depth += 1,
                ')' => {
                    max_depth = depth.max(max_depth);
                    depth -= 1;
                }
                _ => {}
            }
        }
        max_depth
    }
    
    pub fn min_time_to_visit_all_points(points: Vec<Vec<i32>>) -> i32 {
        let mut step = 0;
        let mut index = 1;
        while index < points.len() {
            let x = points[index][0] - points[index - 1][0];
            let y = points[index][1] - points[index - 1][1];
            step += x.abs().max(y.abs());
            index += 1;
        }
        step
    }
    
    pub fn sum_odd_length_subarrays(arr: Vec<i32>) -> i32 {
        let mut d = (arr.len() as i32 + 1) / 2;
        let mut prev = d;
        let mut ret = 0;

        for i in 0..(arr.len() / 2) {
            let j = arr.len() - 1 - i;

            ret += (arr[i] + arr[j]) * prev;
            d -= match arr.len() % 2 {
                0 => 1,
                _ => 2 * (1 - i as i32 % 2),
            };
            prev += d;
        }

        if arr.len() % 2 == 1 {
            ret += arr[arr.len() / 2] * prev;
        }

        ret
    }
    
    pub fn find_numbers(nums: Vec<i32>) -> i32 {
        let mut count = 0;
        for num in nums {
            match num.to_string().len() {
                n if n % 2 == 0 => count += 1,
                _ => (),
            }
        }
        count
    }

    pub fn get_decimal_value(head: Option<Box<ListNode>>) -> i32 {
        let mut sum = 0;
        let mut next = &head;
        while let Some(node) = next{
            sum = sum << 1;
            sum |= node.val;
            next = &node.next;
        }
        sum
    }
    
    pub fn diagonal_sum(mat: Vec<Vec<i32>>) -> i32 {
        let count = (mat.len() + 1) / 2;
        let (mut d, mut nd, mut num) = (0, 0, 0);
        while d < count {
            nd = mat.len() - 1 - d;
            num += mat[d][d] 
                + mat[d][nd] 
                + mat[nd][d]
                + mat[nd][nd];
            d += 1;
        }
        if nd != d {
            num -= 3 * mat[nd][nd]
        }
        num
    }
    
    pub fn busy_student(start_time: Vec<i32>, end_time: Vec<i32>, query_time: i32) -> i32 {
        let mut num = 0;
        for i in 0..start_time.len() {
            if query_time >= start_time[i] && query_time <= end_time[i]{
                num += 1;
            }
        }
        num
        // start_time
        // .iter()
        // .zip(end_time.iter())
        // .filter(|(&s, &e)| s<=query_time&&e>=query_time)
        // .count() as i32
        
        // start_time.drain(..).zip(end_time.drain(..)).fold(0,|s,(st,ed)|s+(st<=query_time&&query_time<=ed) as i32)
    }

    pub fn calculate(s: String) -> i32 {
        let (mut x, mut y) = (1, 0);
        for c in s.chars() {
            match c {
                'A' => x = 2 * x + y,
                'B' => y = 2 * y + x,
                _ => ()
            }
        }
        x + y
    }
    
    pub fn restore_string(mut s: String, indices: Vec<i32>) -> String {
        // let mut t = Vec::new();
        // t.extend(s.chars().zip(indices));
        
        //let mut t = s.drain(..).enumerate().collect::<Vec<(usize,char)>>();

        let mut t: Vec<(char, i32)> = s.drain(..)
                                        .zip(indices.into_iter())
                                        .collect();

        t.sort_by_key(|item| item.1);

        // t.into_iter().map(|(n, _)| n).collect()
        // s.drain(..).unzip::<usize,char,Vec<usize>,String>().1
        t.drain(..).unzip::<char,i32,String,Vec<i32>>().0
    }

    pub fn create_target_array(nums: Vec<i32>, index: Vec<i32>) -> Vec<i32> {
        let mut res:Vec<i32> = vec![];
        // for (i, n) in nums.into_iter().enumerate() {
        for i in 0..nums.len() {
            res.insert(index[i] as usize, nums[i]);
        }
        res
    }

    pub fn balanced_string_split(mut s: String) -> i32 {
        let mut count = 0;
        s.drain(..).filter(|c| {
            match c {
                'L' => count += 1,
                'R' => count -= 1,
                _ => ()
            }
            count == 0
        }).count() as i32
    }
}


#[cfg(test)]
mod tests {
    use super::datastructure;
    use super::Solution;

    #[test]
    fn code1221(){
        // 1221. 分割平衡字符串
        // 在一个「平衡字符串」中，'L' 和 'R' 字符的数量是相同的。
        // 给出一个平衡字符串 s，请你将它分割成尽可能多的平衡字符串。
        // 返回可以通过分割得到的平衡字符串的最大数量。

        assert_eq!(
            Solution::balanced_string_split(String::from("RLRRLLRLRL")),
            4
        );
        assert_eq!(
            Solution::balanced_string_split(String::from("RLLLLRRRLR")),
            3
        );
        assert_eq!(
            Solution::balanced_string_split(String::from("LLLLRRRR")),
            1
        );
    }

    #[test]
    fn code1389(){
        // 1389. 按既定顺序创建目标数组
        // 给你两个整数数组 nums 和 index。
        // 你需要按照以下规则创建目标数组：
        // 目标数组 target 最初为空。
        // 按从左到右的顺序依次读取 nums[i] 和 index[i]，
        // 在 target 数组中的下标 index[i] 处插入值 nums[i] 。
        // 重复上一步，直到在 nums 和 index 中都没有要读取的元素。
        assert_eq!(
            Solution::create_target_array(vec![0,1,2,3,4], vec![0,1,2,2,1]),
            vec![0,4,1,3,2]
        );

        assert_eq!(
            Solution::create_target_array(vec![1,2,3,4,0], vec![0,1,2,3,0]),
            vec![0,1,2,3,4]
        );

        assert_eq!(
            Solution::create_target_array(vec![1], vec![0]),
            vec![1]
        );
    }

    #[test]
    fn code1528(){
        // 1528. 重新排列字符串
        // 给你一个字符串 s 和一个 长度相同 的整数数组 indices 。
        // 请你重新排列字符串 s ，其中第 i 个字符需要移动到 indices[i] 指示的位置。
        // 返回重新排列后的字符串。
        
        assert_eq!(
            Solution::restore_string(String::from("codeleet"), vec![4,5,6,7,0,2,1,3]),
            String::from("leetcode")
        );

        assert_eq!(
            Solution::restore_string(String::from("abc"), vec![0,1,2]),
            String::from("abc")
        );

        assert_eq!(
            Solution::restore_string(String::from("aiohn"), vec![3,1,4,2,0]),
            String::from("nihao")
        );
        assert_eq!(
            Solution::restore_string(String::from("aaiougrt"), vec![4,0,2,6,7,3,1,5]),
            String::from("arigatou")
        );
        assert_eq!(
            Solution::restore_string(String::from("art"), vec![1,0,2]),
            String::from("rat")
        );
    }

    #[test]
    fn codelcp17(){
        // LCP 17. 速算机器人
        // 小扣在秋日市集发现了一款速算机器人。
        // 店家对机器人说出两个数字（记作 x 和 y），请小扣说出计算指令：
        // "A" 运算：使 x = 2 * x + y；
        // "B" 运算：使 y = 2 * y + x。
        // 在本次游戏中，店家说出的数字为 x = 1 和 y = 0，
        // 小扣说出的计算指令记作仅由大写字母 A、B 组成的字符串 s，
        // 字符串中字符的顺序表示计算顺序，请返回最终 x 与 y 的和为多少。

        assert_eq!(
            Solution::calculate(String::from("AB")),
            4
        );
    }

    #[test]
    fn code1450(){
        // 1450. 在既定时间做作业的学生人数
        // 给你两个整数数组 startTime（开始时间）和 endTime（结束时间），
        // 并指定一个整数 queryTime 作为查询时间。
        // 已知，第 i 名学生在 startTime[i] 时开始写作业并于 endTime[i] 时完成作业。
        // 请返回在查询时间 queryTime 时正在做作业的学生人数。
        // 形式上，返回能够使 queryTime 处于区间 [startTime[i], endTime[i]]（含）的学生人数。
        assert_eq!(
            Solution::busy_student(vec![1,2,3], vec![3,2,7], 4),
            1
        );
        assert_eq!(
            Solution::busy_student(vec![4], vec![4], 4),
            1
        );
        assert_eq!(
            Solution::busy_student(vec![4], vec![4], 5),
            0
        );
        assert_eq!(
            Solution::busy_student(vec![1,1,1,1], vec![1,3,2,4], 7),
            0
        );
        assert_eq!(
            Solution::busy_student(vec![9,8,7,6,5,4,3,2,1], vec![10,10,10,10,10,10,10,10,10], 5),
            5
        );
    }

    #[test]
    fn code1572(){
        // 1572. 矩阵对角线元素的和
        // 给你一个正方形矩阵 mat，
        // 请你返回矩阵对角线元素的和。
        // 请你返回在矩阵主对角线上的元素和
        // 副对角线上且不在主对角线上元素的和。
        assert_eq!(
            Solution::diagonal_sum(
                vec![vec![1,2,3],vec![4,5,6],vec![7,8,9]]),
                25
        );
        assert_eq!(
            Solution::diagonal_sum(
                vec![vec![1,1,1,1],vec![1,1,1,1],vec![1,1,1,1],vec![1,1,1,1]]),
                8
        );
    }
    
    #[test]
    fn code1290() {
        // 1290. 二进制链表转整数
        // 给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。
        // 已知此链表是一个整数数字的二进制表示形式。
        // 请你返回该链表所表示数字的 十进制值 。
        let mut n0 = datastructure::ListNode::new(1);
        let mut n1 = datastructure::ListNode::new(0);
        let n2 = datastructure::ListNode::new(1);
        n1.next = Some(Box::new(n2));
        n0.next = Some(Box::new(n1));
        let head = Some(Box::new(n0));
        
        assert_eq!(
            Solution::get_decimal_value(head),
            5
        );
    }
    
    #[test]
    fn code1285() {
        // 1295. 统计位数为偶数的数字
        // 给你一个整数数组 nums，请你返回其中位数为 偶数 的数字的个数。
        assert_eq!(Solution::find_numbers(vec![12, 345, 2, 6, 7896]), 2);
        assert_eq!(Solution::find_numbers(vec![555, 901, 482, 1771]), 1);
    }
    
    #[test]
    fn code1588() {
        // 1588. 所有奇数长度子数组的和
        // 给你一个正整数数组 arr ，
        // 请你计算所有可能的奇数长度子数组的和。
        // 子数组 定义为原数组中的一个连续子序列。
        // 请你返回 arr 中 所有奇数长度子数组的和 。
        assert_eq!(Solution::sum_odd_length_subarrays(vec![1, 4, 2, 5, 3]), 58);
        assert_eq!(Solution::sum_odd_length_subarrays(vec![1, 2]), 3);
        assert_eq!(Solution::sum_odd_length_subarrays(vec![10, 11, 12]), 66);
    }
    
    //--------------2020/11/4--------------
    #[test]
    fn code1266() {
        // 1266. 访问所有点的最小时间
        // 平面上有 n 个点，点的位置用整数坐标表示 points[i] = [xi, yi]。
        // 请你计算访问所有这些点需要的最小时间（以秒为单位）。
        // 你可以按照下面的规则在平面上移动：
        // 每一秒沿水平或者竖直方向移动一个单位长度，
        // 或者跨过对角线（可以看作在一秒内向水平和竖直方向各移动一个单位长度）。
        // 必须按照数组中出现的顺序来访问这些点。
        assert_eq!(
            Solution::min_time_to_visit_all_points(vec![vec![1, 1], vec![3, 4], vec![-1, 0]]),
            7
        );
        assert_eq!(
            Solution::min_time_to_visit_all_points(vec![vec![3, 2], vec![-2, 2]]),
            5
        );
    }
    
    #[test]
    fn code1614() {
        // 1614. 括号的最大嵌套深度
        // 如果字符串满足一下条件之一，则可以称之为
        // 有效括号字符串（valid parentheses string，可以简写为 VPS）：
        // 字符串是一个空字符串 ""，或者是一个不为 "(" 或 ")" 的单字符。
        // 字符串可以写为 AB（A 与 B 字符串连接），
        // 其中 A 和 B 都是 有效括号字符串 。
        // 字符串可以写为 (A)，其中 A 是一个 有效括号字符串 。
        assert_eq!(Solution::max_depth(String::from("(1+(2*3)+((8)/4))+1")), 3);

        assert_eq!(Solution::max_depth(String::from("(1)+((2))+(((3)))")), 3);

        assert_eq!(Solution::max_depth(String::from("1+(2*3)/(2-1)")), 1);

        assert_eq!(Solution::max_depth(String::from("1")), 0);
    }
    //--------------2020/11/3--------------
    #[test]
    fn code1313() {
        // 1313. 解压缩编码列表
        // 给你一个以行程长度编码压缩的整数列表 nums 。
        // 考虑每对相邻的两个元素 [freq, val] = [nums[2*i],
        // nums[2*i+1]] （其中 i >= 0 ），
        // 每一对都表示解压后子列表中有 freq 个值为 val 的元素，
        // 你需要从左到右连接所有子列表以生成解压后的列表。
        // 请你返回解压后的列表。
        assert_eq!(
            Solution::decompress_rl_elist(vec![1, 2, 3, 4]),
            vec![2, 4, 4, 4]
        );
        assert_eq!(
            Solution::decompress_rl_elist(vec![1, 1, 2, 3]),
            vec![1, 3, 3]
        );
    }

    #[test]
    fn code1365() {
        // 1365. 有多少小于当前数字的数字
        // 给你一个数组 nums，对于其中每个元素 nums[i]，
        // 请你统计数组中比它小的所有数字的数目。
        // 换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，
        // 其中 j 满足 j != i 且 nums[j] < nums[i] 。
        // 以数组形式返回答案。
        assert_eq!(
            Solution::smaller_numbers_than_current(vec![8, 1, 2, 2, 3]),
            vec![4, 0, 1, 1, 3]
        );
        assert_eq!(
            Solution::smaller_numbers_than_current(vec![6, 5, 4, 8]),
            vec![2, 1, 0, 3]
        );
        assert_eq!(
            Solution::smaller_numbers_than_current(vec![7, 7, 7, 7]),
            vec![0, 0, 0, 0]
        );
    }
    
    #[test]
    fn code1281() {
        // 1281. 整数的各位积和之差
        // 给你一个整数 n，请你帮忙计算并返回
        // 该整数「各位数字之积」与「各位数字之和」的差。
        assert_eq!(Solution::subtract_product_and_sum(234), 15);

        assert_eq!(Solution::subtract_product_and_sum(4421), 21);
    }

    #[test]
    fn codelcp06() {
        // LCP 06. 拿硬币
        // 桌上有 n 堆力扣币，每堆的数量保存在数组 coins 中。
        // 我们每次可以选择任意一堆，拿走其中的一枚或者两枚，求拿完所有力扣币的最少次数。
        assert_eq!(Solution::min_count(vec![4, 2, 1]), 4);
        assert_eq!(Solution::min_count(vec![2, 3, 10]), 8);
    }

    #[test]
    fn code1108() {
        // 1108. IP 地址无效化
        // 给你一个有效的 IPv4 地址 address，返回这个 IP 地址的无效化版本。
        // 所谓无效化 IP 地址，其实就是用 "[.]" 代替了每个 "."。
        assert_eq!(
            Solution::defang_i_paddr(String::from("1.1.1.1")),
            String::from("1[.]1[.]1[.]1")
        );
    }
    //--------------2020/11/2--------------
    #[test]
    fn code1485() {
        // 1486. 数组异或操作
        // 给你两个整数，n 和 start 。
        // 数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）
        // 且 n == nums.length 。
        // 请返回 nums 中所有元素按位异或（XOR）后得到的结果。
        assert_eq!(Solution::xor_operation(5, 0), 8);
        assert_eq!(Solution::xor_operation(4, 3), 8);
        assert_eq!(Solution::xor_operation(1, 7), 7);
        assert_eq!(Solution::xor_operation(10, 5), 2);
    }
    
    #[test]
    fn offer58ii() {
        // 剑指 Offer 58 - II. 左旋转字符串
        // 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
        // 请定义一个函数实现字符串左旋转操作的功能。
        // 比如，输入字符串"abcdefg"和数字2，
        // 该函数将返回左旋转两位得到的结果"cdefgab"。
        assert_eq!(
            Solution::reverse_left_words(String::from("abcdefg"), 2),
            String::from("cdefgab")
        );
        assert_eq!(
            Solution::reverse_left_words(String::from("lrloseumgh"), 6),
            String::from("umghlrlose")
        );
    }
    
    #[test]
    fn code1603() {
        // 1603. 设计停车系统
        // 请你给一个停车场设计一个停车系统。
        // 停车场总共有三种不同大小的车位：大，中和小，每种尺寸分别有固定数目的车位。
        // 请你实现 ParkingSystem 类：
        // ParkingSystem(int big, int medium, int small)
        // 初始化 ParkingSystem 类，三个参数分别对应每种停车位的数目。
        // bool addCar(int carType)
        // 检查是否有 carType 对应的停车位。
        // carType 有三种类型：大，中，小，分别用数字 1， 2 和 3 表示。
        // 一辆车只能停在  carType 对应尺寸的停车位中。
        // 如果没有空车位，请返回 false ，否则将该车停入车位并返回 true 。

        let mut park = datastructure::ParkingSystem::new(1, 1, 0);
        assert_eq!(park.add_car(1), true);

        assert_eq!(park.add_car(2), true);

        assert_eq!(park.add_car(3), false);

        assert_eq!(park.add_car(1), false);
    }
    
    #[test]
    fn code771() {
        // 771. 宝石与石头
        // 给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。
        // S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
        // J 中的字母不重复，J 和 S中的所有字符都是字母。
        // 字母区分大小写，因此"a"和"A"是不同类型的石头。
        assert_eq!(
            Solution::num_jewels_in_stones(String::from("aA"), String::from("aAAbbbb")),
            3
        );

        assert_eq!(
            Solution::num_jewels_in_stones(String::from("z"), String::from("ZZ")),
            0
        );
    }
    
    #[test]
    fn code1470() {
        // 1470. 重新排列数组
        // 给你一个数组 nums ，数组中有 2n 个元素，
        // 按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。
        // 请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，
        // 返回重排后的数组。
        assert_eq!(
            Solution::shuffle(vec![2, 5, 1, 3, 4, 7], 3),
            vec![2, 3, 5, 4, 1, 7]
        );
        assert_eq!(
            Solution::shuffle(vec![1, 2, 3, 4, 4, 3, 2, 1], 4),
            vec![1, 4, 2, 3, 3, 2, 4, 1]
        );
        assert_eq!(Solution::shuffle(vec![1, 1, 2, 2], 2), vec![1, 2, 1, 2]);
    }
    
    #[test]
    fn code1431() {
        // 1431. 拥有最多糖果的孩子
        // 给你一个数组 candies 和一个整数 extraCandies ，
        // 其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
        // 对每一个孩子，检查是否存在一种方案，
        // 将额外的 extraCandies 个糖果分配给孩子们之后，
        // 此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目。

        assert_eq!(
            Solution::kids_with_candies(vec![2, 3, 5, 1, 3], 3),
            vec![true, true, true, false, true]
        );
        assert_eq!(
            Solution::kids_with_candies(vec![4, 2, 1, 1, 2], 1),
            vec![true, false, false, false, false]
        );
        assert_eq!(
            Solution::kids_with_candies(vec![12, 1, 12], 10),
            vec![true, false, true]
        );
    }
    
    #[test]
    fn code1512() {
        // 1512. 好数对的数目
        // 给你一个整数数组 nums 。
        // 如果一组数字 (i,j) 满足 nums[i] == nums[j] 且 i < j ，
        // 就可以认为这是一组 好数对 。
        // 返回好数对的数目。

        assert_eq!(Solution::num_identical_pairs(vec![1, 2, 3, 1, 1, 3]), 4);
        assert_eq!(Solution::num_identical_pairs(vec![1, 1, 1, 1]), 6);
        assert_eq!(Solution::num_identical_pairs(vec![1, 2, 3]), 0);
    }
}
