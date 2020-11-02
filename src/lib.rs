use std::collections::HashSet;
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
        let mut res:Vec<bool> = vec![true; candies.len()];
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
            res ^= start + 2*i;
            i += 1;
            n -= 1;
        }
        res
    }
}


#[cfg(test)]
mod tests {
    use super::Solution;
    //--------------2020/11/2--------------
    #[test]
    fn code1485(){
        // 1486. 数组异或操作
        // 给你两个整数，n 和 start 。
        // 数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）
        // 且 n == nums.length 。
        // 请返回 nums 中所有元素按位异或（XOR）后得到的结果。
        assert_eq!(
            Solution::xor_operation(5, 0),
            8
        );
        assert_eq!(
            Solution::xor_operation(4, 3),
            8
        );
        assert_eq!(
            Solution::xor_operation(1, 7),
            7
        );
        assert_eq!(
            Solution::xor_operation(10, 5),
            2
        );
 
    }
    #[test]
    fn offer58ii(){
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
    fn code1603(){
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
        struct ParkingSystem {
            capacity: Vec<i32>
        }
        
        impl ParkingSystem {
        
            fn new(big: i32, medium: i32, small: i32) -> Self {
                Self {capacity: vec![big, medium, small]}
            }
            
            fn add_car(&mut self, car_type: i32) -> bool {
                let n = car_type as usize - 1;
                if self.capacity[n] > 0 { self.capacity[n] -= 1; true } else { false }
            }
        }

        let mut park = ParkingSystem::new(1, 1, 0);
        assert_eq!(
            park.add_car(1),
            true
        );

        assert_eq!(
            park.add_car(2),
            true
        );

        assert_eq!(
            park.add_car(3),
            false
        );

        assert_eq!(
            park.add_car(1),
            false
        );
    }
    #[test]
    fn code771(){
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
    fn code1470(){
        // 1470. 重新排列数组
        // 给你一个数组 nums ，数组中有 2n 个元素，
        // 按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。
        // 请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，
        // 返回重排后的数组。
        assert_eq!(
            Solution::shuffle(vec![2,5,1,3,4,7], 3),
            vec![2,3,5,4,1,7] 
        );
        assert_eq!(
            Solution::shuffle(vec![1,2,3,4,4,3,2,1], 4),
            vec![1,4,2,3,3,2,4,1]
        );
        assert_eq!(
            Solution::shuffle(vec![1,1,2,2], 2),
            vec![1,2,1,2]
        );
    }
    #[test]
    fn code1431(){
        // 1431. 拥有最多糖果的孩子
        // 给你一个数组 candies 和一个整数 extraCandies ，
        // 其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
        // 对每一个孩子，检查是否存在一种方案，
        // 将额外的 extraCandies 个糖果分配给孩子们之后，
        // 此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目。

        assert_eq!(
            Solution::kids_with_candies(vec![2, 3, 5, 1, 3], 3),
            vec![true,true,true,false,true]
        );
        assert_eq!(
            Solution::kids_with_candies(vec![4,2,1,1,2], 1),
            vec![true,false,false,false,false]
        );
        assert_eq!(
            Solution::kids_with_candies(vec![12,1,12], 10),
            vec![true,false,true]
        );
    }
    #[test]
    fn code1512() {
        // 1512. 好数对的数目
        // 给你一个整数数组 nums 。
        // 如果一组数字 (i,j) 满足 nums[i] == nums[j] 且 i < j ，
        // 就可以认为这是一组 好数对 。
        // 返回好数对的数目。

 
        assert_eq!(
            Solution::num_identical_pairs(vec![1,2,3,1,1,3]),
            4
        );
        assert_eq!(
            Solution::num_identical_pairs(vec![1,1,1,1]),
            6
        );
        assert_eq!(
            Solution::num_identical_pairs(vec![1,2,3]),
            0
        );
    }
}