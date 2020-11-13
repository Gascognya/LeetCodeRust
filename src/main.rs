// fn main(){
//     let mut v = vec![1, 2, 3];
//     println!("{:?}", v.drain(..));
//     // Drain([1, 2, 3])

//     let mut s = String::from("codeleet");
//     let s = s.drain(..);
//     println!("{:?}", s);
//     // 得到Drain { .. }   

//     let s = s.enumerate();
//     println!("{:?}", s);
//     // Enumerate { iter: Drain { .. }, count: 0 }

//     let s = s.collect::<Vec<(usize,char)>>();
//     println!("{:?}", s);
//     // [(usize, char), (usize, char)]


//     // use std::collections::HashMap;
//     // let mut m: HashMap<u32, &str> = HashMap::new();
//     // m.insert(1, "hello");
//     // m.insert(2, "world");
//     // let v: Vec<u32> = vec![1,2,3];
//     // let b: Vec<(u32, &str)> = v 
//     //     .iter()
//     //     .filter_map(|&k| m.get(&k).map(|&v| (k, v)))
//     //     .collect();
//     // for (x,y) in b {
//     //     println!("{},{}", x,y);
//     // }


// }
use std::num::ParseIntError;

fn multiply(first_number_str: &str, second_number_str: &str) -> Result<i32, ParseIntError> {
    let n = 1;
    let s = format!("{num:>count$}", num='9', count=n as usize);
    let first_number = s.parse::<i32>()?;
    let second_number = second_number_str.to_string().parse::<i32>()?;

    Ok(first_number * second_number)
}

fn print(result: Result<i32, ParseIntError>) {
    match result {
        Ok(n)  => println!("n is {}", n),
        Err(e) => println!("Error: {}", e),
    }
}

fn main() {
    print(multiply("10", "2"));
    print(multiply("t", "2"));
}