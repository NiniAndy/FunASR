# Git融合方法

#### 1.确认当前分支

```
git checkout mymerge
```

#### 2.更新本地代码到github版本

```
git pull origin mymerge
```

#### 3.从github中拉取所有branch信息

```
git fetch origin
```

#### 4.选择要和哪个本版进行融合

```
git merge origin/main
```

#### 5.冲突判断

#### 6.确认完成冲突合并

```
 git commit
```

#### 7.上传合并后文件

```
git push origin mymerge
```

