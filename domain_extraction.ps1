#need the CATHSOLID.txt file in the same directory as this script
$data = Get-Content CATHSOLID.txt | ForEach-Object {
    $parts = $_ -split "\s+"
    [PSCustomObject]@{ ID = $parts[0]; Class = $parts[1] }
}
# Sample 6000 IDs per class and save to text files 
$data | Where-Object { $_.Class -eq "1" } | Get-Random -Count 6000 | ForEach-Object { $_.ID } | Set-Content alpha_sample.csv
$data | Where-Object { $_.Class -eq "2" } | Get-Random -Count 6000 | ForEach-Object { $_.ID } | Set-Content beta_sample.csv
$data | Where-Object { $_.Class -eq "3" } | Get-Random -Count 6000 | ForEach-Object { $_.ID } | Set-Content alpha_beta_sample.csv
$data | Where-Object { $_.Class -eq "4" } | Get-Random -Count 6000 | ForEach-Object { $_.ID } | Set-Content fewss_sample.csv